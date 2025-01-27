from zipfile import ZipFile
from pathlib import Path
import subprocess
import pandas as pd
from rich.progress import track
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import psutil


DATA_DIR = Path("/home/rjslater/Documents/Projects/UCR6-Stats-Bot/data")
DECODER_PATH = Path("/home/rjslater/Documents/Projects/UCR6-Stats-Bot/src/decoder/r6-dissect/r6-dissect")


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
                team_name = teams[player["teamIndex"]] if teams[player["teamIndex"]] not in {"YOUR TEAM", "OPPONENTS"} else ""
                player_dict = {"player_id": player["profileID"], "player_name": player["username"], "team_name": team_name}
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
            players = sorted([player["profileID"] for player in round["players"] if player["teamIndex"] == team_index])
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
                    unique_team["player_ids"] = list(set(unique_team["player_ids"]) | set(team["player_ids"]))
                    # Track name usage
                    unique_team["name_count"][team["team_name"]] += 1
                    team_matched = True
                    break

            if not team_matched:
                # Add a new unique team
                unique_teams.append({"name_count": Counter([team["team_name"]]), "player_ids": team["player_ids"]})

        # Finalize the unique teams with their most frequent names
        for unique_team in unique_teams:
            # Pick the most frequent non-empty name
            most_frequent_name = max(
                (name for name in unique_team["name_count"] if name.strip()), key=lambda x: unique_team["name_count"][x], default=""
            )

            # Guess team name if none is available
            if not most_frequent_name.strip():
                # Convert player IDs to names and count tags
                player_names = [get_player_name(player_id) for player_id in unique_team["player_ids"]]
                player_tags = [extract_tag(player_name) for player_name in player_names if extract_tag(player_name)]
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
        player_info_df["team_ids"] = player_info_df["team_ids"].apply(lambda x: [] if pd.isna(x) else x)

    # Initial player table
    player_info_df = pd.DataFrame()
    for match_json_path in Path(data_dir / "2_decoded_replays/").glob("*.json"):
        player_info_df = pd.concat((player_info_df, get_players(match_json_path)), ignore_index=True)
    player_info_df.drop_duplicates(subset="player_id", inplace=True)
    player_info_df.sort_values(["team_name", "player_name"], inplace=True)

    # Initial team table
    team_instances = []
    for match_json_path in Path(data_dir / "2_decoded_replays").rglob("*.json"):
        team_instances.extend(get_teams(match_json_path))
    unique_teams = condense_teams(team_instances, get_players(match_json_path))
    team_info_df = pd.DataFrame(unique_teams, columns=["team_id", "team_name", "player_ids"])

    # Update player table with team info
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
                "round_number": i + 1,
                "game_version": round["gameVersion"],
                "code_version": round["codeVersion"],
                "timestamp": round["timestamp"],
                "match_type": round["matchType"]["name"],
                "map_name": round["map"]["name"],
                "site": round["site"],
                "rounds_per_match": round["roundsPerMatch"],
                "rounds_per_match_overtime": round["roundsPerMatchOvertime"],
                "team_0_name": round["teams"][0]["name"],
                # TODO: Score is bugged
                "team_0_score_before": round["teams"][0]["startingScore"],
                "team_0_score_after": round["teams"][0]["score"],
                "team_0_won_round": round["teams"][0]["won"],
                "team_0_side": round["teams"][0]["role"],
                "team_1_name": round["teams"][1]["name"],
                "team_1_score_before": round["teams"][1]["startingScore"],
                "team_1_score_after": round["teams"][1]["score"],
                "team_1_won_round": round["teams"][1]["won"],
                "team_1_side": round["teams"][1]["role"],
                "win_condition": round["teams"][0]["winCondition"] if "winCondition" in round["teams"][0] else round["teams"][1]["winCondition"],
            }

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
                    round_dict["defuser_planted_by"] = get_player_id_from_name(round, event["username"])
                    round_dict["defuser_plant_time"] = event["time"]
                if event["type"]["name"] == "DefuserDisableComplete":
                    round_dict["defuser_disabled"] = True
                    round_dict["defuser_disabled_by"] = get_player_id_from_name(round, event["username"])
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

        def get_player_side_from_player_id(player_dicts: list[dict], team_dicts:list[dict], player_id: str):
            for player in player_dicts:
                if player["profileID"] == player_id:
                    team_index = player["teamIndex"]
                    return team_dicts[team_index]["role"]
            return None

        def find_trade_kill(kill: dict, kills: list[dict]):
            """Trade is defined as someone who kills someone who just got a kill within 10s of the first kill."""
            for other_kill in kills:
                if other_kill["killer_id"] == kill["victim_id"] and other_kill["time_seconds"] - kill["time_seconds"] <= -10:
                    return True
            return False

        with open(match_json_path, "r") as f:
            match_json = json.load(f)

        kills = []
        match_id = match_json_path.stem
        for i, round in enumerate(match_json["rounds"]):
            round_num = i + 1
            round_id = f"{match_json_path.stem}_{i + 1}"

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
                    "killer_operator": get_operator_from_player_id(round["players"], killer_player_id),
                    "victim_operator": get_operator_from_player_id(round["players"], victim_player_id),
                    "killer_side": get_player_side_from_player_id(round["players"], round["teams"], killer_player_id),
                    "victim_side": get_player_side_from_player_id(round["players"], round["teams"], victim_player_id),
                    "killer_spawn": get_spawn_location_from_player_id(round["players"], killer_player_id),
                    "victim_spawn": get_spawn_location_from_player_id(round["players"], victim_player_id),
                }
                kill["trade"] = find_trade_kill(kill, kills)
                kills.append(kill)

        return pd.DataFrame(kills)

    kill_data_df = pd.DataFrame()
    for match_json_path in Path(data_dir / "2_decoded_replays").rglob("*.json"):
        kill_data_df = pd.concat((kill_data_df, get_kills(match_json_path)), ignore_index=True)
    return kill_data_df


def decode_matches(data_dir: Path = DATA_DIR, decoder_path: Path = DECODER_PATH):
    # Step 1: Extract the match zips
    zip_files = list((data_dir / "0_replay_zips").iterdir())
    for match_zip in track(zip_files, description="Extracting match zips", total=len(zip_files)):
        _extract_match_zip(match_zip)

    # Step 2: Decode the match folders
    match_folders = list((data_dir / "1_replay_folders").iterdir())
    futures = []
    thread_count = round(_memory_available() / 4)  # overestimate of 4 GB per thread
    with ThreadPoolExecutor(thread_count) as executor:
        for match_folder in match_folders:
            futures.append(executor.submit(_decode_match_folder, match_folder, decoder_path))
        for future in track(as_completed(futures), total=len(futures), description=f"Decoding match folders ({thread_count})"):
            future.result()

    # Step 3: Make the tables we want
    player_info_df, team_info_df = generate_players_and_teams_tables(data_dir)
    round_data_df = generate_rounds_table(data_dir)
    kill_data_df = generate_kill_table(data_dir)

    # Step 4: Save the tables
    player_info_df.to_csv(data_dir / "3_tables/player_info.csv", index=False)
    team_info_df.to_csv(data_dir / "3_tables/teams.csv", index=False)
    round_data_df.to_csv(data_dir / "3_tables/round_data.csv", index=False)
    kill_data_df.to_csv(data_dir / "3_tables/kill_data.csv", index=False)


if __name__ == "__main__":
    decode_matches()