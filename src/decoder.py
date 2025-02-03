from zipfile import ZipFile
from pathlib import Path
import subprocess
import pandas as pd
from rich.progress import track
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import psutil
from ast import literal_eval

from utils import (
    get_game_id_from_match_id,
    get_team_id_from_roster,
    get_team_ids_from_round_dict,
    get_team_name_from_id,
)


DATA_DIR = Path("/home/rjslater/Documents/Projects/UCR6-Stats-Bot/data")
DECODER_PATH = Path("/home/rjslater/Documents/Projects/UCR6-Stats-Bot/src/r6-dissect/r6-dissect")


def _memory_available():
    """Gets total system memory in GB."""
    virtual_memory = psutil.virtual_memory()
    return virtual_memory.total / (1024**3)


def get_player_id_from_name(round: dict, player_name: str):
    for player in round["players"]:
        if player["username"] == player_name:
            return player["profileID"]
    return None


def _extract_match_zip(match_zip: Path, data_dir: Path = DATA_DIR):
    with ZipFile(match_zip, "r") as zip_ref:
        zip_ref.extractall(data_dir / "1_replay_folders")


def _decode_match_folder(match_dir: Path, decoder_path: Path = DECODER_PATH):
    # If already decoded, skip
    processed_files = [f"{file.stem}.json" for file in (DATA_DIR / "2_decoded_replays").iterdir()]
    if f"{match_dir.stem}.json" in processed_files:
        return

    json_path = f"{(DATA_DIR / '2_decoded_replays') / match_dir.stem}.json"
    command = f"{decoder_path} {match_dir} -o {json_path}"
    subprocess.run(command, shell=True, capture_output=True, text=True)
    with open(json_path, "r") as f:
        data = json.load(f)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)


def generate_players_and_teams_tables(data_dir: Path = DATA_DIR):
    def get_players(match_json_path: Path):
        with open(match_json_path, "r") as f:
            match_json = json.load(f)

        players = []
        for round in match_json["rounds"]:
            teams = [team["name"] for team in round["teams"]]
            for player in round["players"]:
                team_name = (
                    teams[player["teamIndex"]]
                    if teams[player["teamIndex"]] not in {"YOUR TEAM", "OPPONENTS"}
                    else ""
                )
                player_dict = {
                    "player_id": player["profileID"],
                    "player_name": player["username"],
                    "team_name": team_name,
                }
                # If player_dict[id] not in any dict in players, add this new player
                if any(player_dict["player_id"] == p["player_id"] for p in players):
                    continue
                else:
                    players.append(player_dict)

        return pd.DataFrame(players)

    def get_teams(match_json_path: Path):
        with open(match_json_path, "r") as f:
            match_json = json.load(f)

        teams = []
        round = match_json["rounds"][0]
        for team_index, team in enumerate(round["teams"]):
            team_name = team["name"] if team["name"] not in {"YOUR TEAM", "OPPONENTS"} else ""
            players = sorted(
                [
                    player["profileID"]
                    for player in round["players"]
                    if player["teamIndex"] == team_index
                ]
            )
            teams.append({"team_name": team_name, "player_ids": players})

        return teams

    def condense_teams(team_list, player_info_df):
        def are_same_team(players1, players2):
            """Check if two rosters have at least 3 players in common."""
            return len(set(players1) & set(players2)) >= 3

        def extract_tag(player_name):
            """Extract the tag from a player's username (e.g., `Player.TAG`)."""
            if "." in player_name:
                return player_name.split(".")[-1]
            return None

        def get_player_name(player_id):
            """Look up the player name from the DataFrame using the player ID."""
            result = player_info_df.loc[player_info_df["player_id"] == player_id, "player_name"]
            return result.iloc[0] if not result.empty else f"Unknown({player_id})"

        unique_teams = []

        for team in team_list:
            team_matched = False
            for unique_team in unique_teams:
                if are_same_team(team["player_ids"], unique_team["player_ids"]):
                    # Merge players
                    unique_team["player_ids"] = list(
                        set(unique_team["player_ids"]) | set(team["player_ids"])
                    )
                    # Track name usage
                    unique_team["name_count"][team["team_name"]] += 1
                    team_matched = True
                    break

            if not team_matched:
                # Add a new unique team
                unique_teams.append(
                    {"name_count": Counter([team["team_name"]]), "player_ids": team["player_ids"]}
                )

        # Finalize the unique teams with their most frequent names
        for unique_team in unique_teams:
            # Pick the most frequent non-empty name
            most_frequent_name = max(
                (name for name in unique_team["name_count"] if name.strip()),
                key=lambda x: unique_team["name_count"][x],
                default="",
            )

            # Guess team name if none is available
            if not most_frequent_name.strip():
                # Convert player IDs to names and count tags
                player_names = [
                    get_player_name(player_id) for player_id in unique_team["player_ids"]
                ]
                player_tags = [
                    extract_tag(player_name)
                    for player_name in player_names
                    if extract_tag(player_name)
                ]
                tag_counts = Counter(player_tags)
                guessed_name = max(tag_counts, key=tag_counts.get, default="").strip()

                most_frequent_name = f"{guessed_name}" if guessed_name else "Unknown Team"

            unique_team["team_name"] = most_frequent_name
            del unique_team["name_count"]  # Clean up temporary counter

        # Set first column to be integer ID for each time, starting at 0 and counting up by 1
        for i, team in enumerate(unique_teams):
            team["team_id"] = i

        return unique_teams

    def update_players_with_team_info(player_info_df, team_info_df):
        """Add team_ids column and properly assign team IDs to players.

        Note that a player can be on more than one team, so the team ID column is a list.
        """
        player_teams_map = {}  # {player_id: [team_id1, team_id2, ...]}
        player_info_df.drop(columns=["team_name"], inplace=True)
        for team_id, team_row in team_info_df.iterrows():
            for player_id in team_row["player_ids"]:
                if player_id in player_teams_map:
                    player_teams_map[player_id].append(team_id)
                else:
                    player_teams_map[player_id] = [team_id]

        player_info_df["team_ids"] = player_info_df["player_id"].map(player_teams_map)
        # ! Is this line needed?
        # player_info_df["team_ids"] = player_info_df["team_ids"].apply(lambda x: [] if pd.isna(x) else x)

    def update_players_with_all_names(player_info_df: pd.DataFrame, data_dir: Path):
        """Iterate over all matches and find all usernames associated with each player ID."""
        for i, row in player_info_df.iterrows():
            player_id = row["player_id"]
            player_names = []
            # Sorted reverse to make sure most recent name is first in list
            for match_json_path in sorted(
                Path(data_dir / "2_decoded_replays").glob("*.json"), reverse=True
            ):
                with open(match_json_path) as f:
                    match_json = json.load(f)
                for player_dict in match_json["rounds"][0]["players"]:
                    if (
                        player_dict["profileID"] == player_id
                        and player_dict["username"] not in player_names
                    ):
                        player_names.append(player_dict["username"])
            player_info_df.loc[i, "player_name"] = player_names
        player_info_df.rename(columns={"player_name": "player_names"}, inplace=True)

    # Initial player table
    player_info_df = pd.DataFrame()
    for match_json_path in Path(data_dir / "2_decoded_replays/").glob("*.json"):
        player_info_df = pd.concat(
            (player_info_df, get_players(match_json_path)), ignore_index=True
        )
    player_info_df.drop_duplicates(subset="player_id", inplace=True)
    player_info_df.sort_values(["team_name", "player_name"], inplace=True)

    # Initial team table
    team_instances = []
    for match_json_path in Path(data_dir / "2_decoded_replays").rglob("*.json"):
        team_instances.extend(get_teams(match_json_path))
    unique_teams = condense_teams(team_instances, get_players(match_json_path))
    team_info_df = pd.DataFrame(unique_teams, columns=["team_id", "team_name", "player_ids"])

    # Update player table with all names and team info
    update_players_with_all_names(player_info_df, data_dir)
    update_players_with_team_info(player_info_df, team_info_df)

    return player_info_df, team_info_df


# TODO: Score is bugged
def generate_rounds_table(data_dir: Path = DATA_DIR):
    def get_round_info(match_json_path: Path):
        with open(match_json_path, "r") as f:
            match_json = json.load(f)

        rounds = []
        for i, round in enumerate(match_json["rounds"]):
            round_dict = {
                "match_id": match_json_path.stem,
                "round_id": f"{match_json_path.stem}_{i + 1}",
                "round_num": i + 1,
                "game_version": round["gameVersion"],
                "code_version": round["codeVersion"],
                "timestamp": round["timestamp"],
                "match_type": round["matchType"]["name"],
                "map_name": round["map"]["name"],
                "site": round["site"],
                "rounds_per_match": round["roundsPerMatch"],
                "rounds_per_match_overtime": round["roundsPerMatchOvertime"],
            }

            # Figure out team IDs from rosters
            team_0_player_ids = [
                player["profileID"] for player in round["players"] if player["teamIndex"] == 0
            ]
            team_1_player_ids = [
                player["profileID"] for player in round["players"] if player["teamIndex"] == 1
            ]
            team_0_id = get_team_id_from_roster(team_0_player_ids)
            team_1_id = get_team_id_from_roster(team_1_player_ids)

            # Add team-related info
            round_dict["team_0_id"] = team_0_id
            # TODO: Score is bugged
            round_dict["team_0_score_before"] = round["teams"][0]["startingScore"]
            round_dict["team_0_score_after"] = round["teams"][0]["score"]
            round_dict["team_0_won_round"] = round["teams"][0]["won"]
            round_dict["team_0_side"] = round["teams"][0]["role"]
            round_dict["team_1_id"] = team_1_id
            # TODO: Score is bugged
            round_dict["team_1_score_before"] = round["teams"][1]["startingScore"]
            round_dict["team_1_score_after"] = round["teams"][1]["score"]
            round_dict["team_1_won_round"] = round["teams"][1]["won"]
            round_dict["team_1_side"] = round["teams"][1]["role"]

            # Figure out win condition
            win_condition = "ERROR - UNKNOWN"
            if "winCondition" in round["teams"][0]:
                win_condition = round["teams"][0]["winCondition"]
            elif "winCondition" in round["teams"][1]:
                win_condition = round["teams"][1]["winCondition"]
            round_dict["win_condition"] = win_condition

            # By default, the defuser is not planted or defused
            round_dict["defuser_planted"] = False
            round_dict["defuser_planted_by"] = None
            round_dict["defuser_plant_time"] = None
            round_dict["defuser_disabled"] = False
            round_dict["defuser_disabled_by"] = None
            round_dict["defuser_disable_time"] = None

            # Check for defuser plant and defuser disable events
            for event in round["matchFeedback"]:
                if event["type"]["name"] == "DefuserPlantComplete":
                    round_dict["defuser_planted"] = True
                    round_dict["defuser_planted_by"] = get_player_id_from_name(
                        round, event["username"]
                    )
                    round_dict["defuser_plant_time"] = event["time"]
                if event["type"]["name"] == "DefuserDisableComplete":
                    round_dict["defuser_disabled"] = True
                    round_dict["defuser_disabled_by"] = get_player_id_from_name(
                        round, event["username"]
                    )
                    round_dict["defuser_disable_time"] = event["time"]

            rounds.append(round_dict)

        return pd.DataFrame(rounds)

    round_data_df = pd.DataFrame()
    for match_json_path in Path(data_dir / "2_decoded_replays").rglob("*.json"):
        match_round_data_df = get_round_info(match_json_path)
        round_data_df = pd.concat((round_data_df, match_round_data_df), ignore_index=True)
    return round_data_df


def generate_kill_table(data_dir: Path = DATA_DIR):
    def get_kills(match_json_path: Path):
        def get_operator_from_player_id(player_dicts: list[dict], player_id: str):
            for player in player_dicts:
                if player["profileID"] == player_id:
                    return player["operator"]["name"]
            return None

        def get_spawn_location_from_player_id(player_dicts: list[dict], player_id: str):
            for player in player_dicts:
                if player["profileID"] == player_id:
                    return player["spawn"]
            return None

        def get_player_side_from_player_id(
            player_dicts: list[dict], team_dicts: list[dict], player_id: str
        ):
            for player in player_dicts:
                if player["profileID"] == player_id:
                    team_index = player["teamIndex"]
                    return team_dicts[team_index]["role"]
            return None

        with open(match_json_path, "r") as f:
            match_json = json.load(f)

        kills = []
        match_id = match_json_path.stem
        for i, round in enumerate(match_json["rounds"]):
            round_num = i + 1
            round_id = f"{match_json_path.stem}_{i + 1}"

            round_kills = []  # Store all kills for this round before processing trades

            for event in round["matchFeedback"]:
                if event["type"]["name"] != "Kill":
                    continue

                killer_player_id = get_player_id_from_name(round, event["username"])
                victim_player_id = get_player_id_from_name(round, event["target"])

                kill = {
                    "match_id": match_id,
                    "round_id": round_id,
                    "round_num": round_num,
                    "killer_id": killer_player_id,
                    "victim_id": victim_player_id,
                    "headshot": event["headshot"],
                    "time": event["time"],
                    "time_seconds": event["timeInSeconds"],
                    "map": round["map"]["name"],
                    "killer_operator": get_operator_from_player_id(
                        round["players"], killer_player_id
                    ),
                    "victim_operator": get_operator_from_player_id(
                        round["players"], victim_player_id
                    ),
                    "killer_side": get_player_side_from_player_id(
                        round["players"], round["teams"], killer_player_id
                    ),
                    "victim_side": get_player_side_from_player_id(
                        round["players"], round["teams"], victim_player_id
                    ),
                    "killer_spawn": get_spawn_location_from_player_id(
                        round["players"], killer_player_id
                    ),
                    "victim_spawn": get_spawn_location_from_player_id(
                        round["players"], victim_player_id
                    ),
                    "kill_is_trade": False,  # Placeholder
                    "death_was_traded": False,  # Placeholder
                }

                round_kills.append(kill)

            # Now that we have all kills for the round, process trades
            for kill in round_kills:
                for other_kill in round_kills:
                    if (
                        other_kill["killer_id"] == kill["victim_id"]  # The victim later gets a kill
                        and 0
                        <= other_kill["time_seconds"] - kill["time_seconds"]
                        <= 10  # Within 10s
                    ):
                        kill["kill_is_trade"] = True

                        # Find the death event for the traded victim and mark it
                        for victim_kill in round_kills:
                            if (
                                victim_kill["victim_id"] == other_kill["killer_id"]
                            ):  # The traded player
                                other_kill["death_was_traded"] = True

            kills.extend(round_kills)  # Append all processed kills for this round

        return pd.DataFrame(kills)

    kill_data_df = pd.DataFrame()
    for match_json_path in Path(data_dir / "2_decoded_replays").rglob("*.json"):
        kill_data_df = pd.concat((kill_data_df, get_kills(match_json_path)), ignore_index=True)
    return kill_data_df.sort_values(
        by=["match_id", "round_num", "time_seconds"], ascending=[True, True, False]
    )


def update_op_bans(data_dir: Path = DATA_DIR):
    op_bans_csv = data_dir / "3_tables/op_bans.csv"
    op_bans_df = pd.read_csv(op_bans_csv)
    for match_json in (data_dir / "2_decoded_replays").glob("*.json"):
        match_id = match_json.stem
        if match_id in list(op_bans_df["match_id"]):
            continue
        with open(match_json) as f:
            match_dict = json.load(f)
            team_0_id, team_1_id = get_team_ids_from_round_dict(match_dict["rounds"][0])
        match_data_dict = {
            "game_id": [get_game_id_from_match_id(match_id)],
            "match_id": [match_id],
            "map": match_dict["rounds"][0]["map"]["name"],
            "team_0_id": [team_0_id],
            "team_1_id": [team_1_id],
        }
        op_bans_df = pd.concat((op_bans_df, pd.DataFrame(match_data_dict)))
    op_bans_df = op_bans_df.sort_values(by=["game_id", "match_id"])
    op_bans_df.to_csv(op_bans_csv, index=False)


def update_map_bans(data_dir: Path = DATA_DIR):
    def get_team_ids_from_match_id(match_id: str, data_dir: Path = DATA_DIR):
        df = pd.read_csv(data_dir / "3_tables/round_data.csv")
        df = df[df["match_id"] == match_id]
        return (df.iloc[0]["team_0_id"], df.iloc[0]["team_1_id"])

    game_info_df = pd.read_csv(data_dir / "3_tables/game_info.csv")
    map_ban_df = pd.read_csv(data_dir / "3_tables/map_bans.csv")
    # Add all new games to map bans df and generate template to make it easier to manually fill out
    for _, row in game_info_df.iterrows():
        game_id = row["game_id"]
        # If game is already in map bans or isn't a match, skip
        if game_id in map_ban_df["game_id"].unique() or row["game_type"] == "scrim":
            continue

        # Determine pick/ban order from game format
        game_format = row["format"]
        pick_ban_order = ("ban", "ban", "ban", "ban", "pick", "pick", "ban", "ban", "pick")  # Bo3
        if game_format == "Bo1":
            pick_ban_order = ("ban", "ban", "ban", "ban", "ban", "ban", "ban", "ban", "pick")
        elif game_format == "Bo5":
            pick_ban_order = ("ban", "ban", "pick", "pick", "ban", "ban", "pick", "pick", "pick")

        # Figure out which teams are playing
        match_id = literal_eval(row["match_ids"])[0]
        if (
            "," not in row["match_ids"]
        ):  # There's only 1 match ID so no commma so no tuple so it gets parsed as string
            match_id = row["match_ids"].removeprefix("('").removesuffix("')")
        team_ids = list(get_team_ids_from_match_id(match_id))
        team_names = [get_team_name_from_id(team_ids[0]), get_team_name_from_id(team_ids[1])]

        # Ask user for info
        print(
            f"\n{row['league']} - ({team_ids[0]}) {team_names[0]} vs ({team_ids[1]}) {team_names[1]}"
        )
        response = input("Do you have map ban info? [Y/n] ").lower().strip()
        if response.startswith("n"):
            print("No map ban info :(")
            continue
        # If we have map ban info, we need to ask the user for it
        response = int(input("Who banned first (team ID)? ").strip())
        if response not in team_ids:
            print("ERROR: invalid team ID")
            exit()
        # Make the first team id the one who banned first
        if response == team_ids[1]:
            team_ids.reverse()
            team_names.reverse()

        # Iterate over pick/ban order and prompt user for maps
        # TODO: this really shouldn't be hardcoded
        map_pool = [
            "Border",
            "Bank",
            "Chalet",
            "Club House",
            "Consulate",
            "Kafe Dostoyevsky",
            "Lair",
            "NightHaven Labs",
            "Skyscraper",
        ]
        for i, veto_type in enumerate(pick_ban_order):
            selected_map = map_pool[0]
            if len(map_pool) > 1:
                print(", ".join(map_pool))
                response = input(f"{team_names[0]} {veto_type}: ").lower().strip()
                for map_name in map_pool:
                    if map_name.lower().startswith(response):
                        selected_map = map_name
                        map_pool.remove(map_name)
                        break

            veto_info = {
                "game_id": game_id,
                "team_id": team_ids[0],
                "veto_number": i,
                "veto_type": veto_type,
                "map": selected_map,
            }
            map_ban_df = pd.concat((map_ban_df, pd.DataFrame([veto_info])), ignore_index=True)

            team_ids.reverse()
            team_names.reverse()
        map_ban_df.to_csv(data_dir / "3_tables/map_bans.csv", index=False)


def check_for_unentered_games(data_dir: Path = DATA_DIR):
    game_info_df = pd.read_csv(data_dir / "3_tables/game_info.csv")
    existing_match_ids = []
    for game_match_ids in game_info_df["match_ids"].values:
        # print(game_match_ids)
        if "," in game_match_ids:
            existing_match_ids.extend(list(literal_eval(game_match_ids)))
        else:
            existing_match_ids.append(game_match_ids.replace("('", "").replace("')", ""))

    # Find if there are any replays not added to game info csv
    missing_count = 0
    for json_file in sorted((data_dir / "2_decoded_replays").glob("*.json")):
        if json_file.stem not in existing_match_ids:
            if missing_count == 0:
                print("Enter game information for the following replays:")
            print(json_file.stem)
            missing_count += 1
    if missing_count:
        input()


def decode_matches(data_dir: Path = DATA_DIR, decoder_path: Path = DECODER_PATH):
    # Step 1: Extract the match zips
    zip_files = list((data_dir / "0_replay_zips").iterdir())
    for match_zip in track(zip_files, description="Extracting match zips", total=len(zip_files)):
        _extract_match_zip(match_zip)

    # Step 2: Decode the match folders
    match_folders = list((data_dir / "1_replay_folders").iterdir())
    futures = []
    thread_count = min(
        len(match_folders), round(_memory_available() / 6)
    )  # overestimate of 6 GB per thread
    with ThreadPoolExecutor(thread_count) as executor:
        for match_folder in match_folders:
            futures.append(executor.submit(_decode_match_folder, match_folder, decoder_path))
        for future in track(
            as_completed(futures),
            total=len(futures),
            description=f"Decoding match folders ({thread_count})",
        ):
            future.result()

    # Step 3: Make the tables we want
    player_info_df, team_info_df = generate_players_and_teams_tables(data_dir)
    player_info_df.to_csv(data_dir / "3_tables/player_info.csv", index=False)
    team_info_df.to_csv(data_dir / "3_tables/teams.csv", index=False)

    round_data_df = generate_rounds_table(data_dir)
    round_data_df.to_csv(data_dir / "3_tables/round_data.csv", index=False)

    kill_data_df = generate_kill_table(data_dir)
    kill_data_df.to_csv(data_dir / "3_tables/kill_data.csv", index=False)

    # Step 4: Post-processing for manual entry fields
    check_for_unentered_games(data_dir)
    update_map_bans(data_dir)
    update_op_bans(data_dir)
    print(f"Reminder to set op bans:\n{DATA_DIR / '3_tables/op_bans.csv'}")


if __name__ == "__main__":
    decode_matches()
