import pandas as pd

from utils import KILL_DATA_CSV, OBJ_DATA_CSV, OP_BAN_CSV, ROUND_DATA_CSV, player_is_on_team


def get_team_winrate(team_id: int, side: str | None = None, map_name: str | None = None):
    df = pd.read_csv(ROUND_DATA_CSV)
    df = df[(df["team_0_id"] == team_id) | (df["team_1_id"] == team_id)]
    if side is not None:
        assert side in {"Attack", "Defense"}
        df = df[
            ((df["team_0_id"] == team_id) & (df["team_0_side"] == side))
            | ((df["team_1_id"] == team_id) & (df["team_1_side"] == side))
        ]
    if map_name is not None:
        assert map_name in df["map_name"].unique()
        df = df[df["map_name"] == map_name]

    win_df = df[
        ((df["team_0_id"] == team_id) & (df["team_0_score_after"] > df["team_0_score_before"]))
        | ((df["team_1_id"] == team_id) & (df["team_1_score_after"] > df["team_1_score_before"]))
    ]

    winrate = len(win_df) / len(df)
    round_count = len(df)
    return winrate, round_count


def _get_matches_team_has_played(team_id: int):
    # TODO: add teams and scores to game info csv and use that here
    df = pd.read_csv(OP_BAN_CSV)
    df = df[(df["team_0_id"] == team_id) | (df["team_1_id"] == team_id)]
    return list(df["match_id"].unique())


def get_team_opening_kill_rate(team_id: int):
    match_ids = _get_matches_team_has_played(team_id)
    df = pd.read_csv(KILL_DATA_CSV)
    df = df[df["match_id"].isin(match_ids)]

    opening_kill_count = 0
    total_opening_kills = 0
    while len(df):
        if player_is_on_team(df.iloc[0]["killer_id"], team_id):
            opening_kill_count += 1
        total_opening_kills += 1

        cur_round_id = df.iloc[0]["round_id"]
        df = df[df["round_id"] != cur_round_id]

    return opening_kill_count / total_opening_kills, total_opening_kills


def get_player_rounds_played(player_id: str, team_id: int | None = None):
    df = pd.read_csv(KILL_DATA_CSV)
    # TODO: handle filtering only by games where player was on a particular team)
    # Filter matches where the player was either a killer or a victim
    player_matches = df[(df["killer_id"] == player_id) | (df["victim_id"] == player_id)]

    # Get the maximum round_num per match
    rounds_per_match = player_matches.groupby("match_id")["round_num"].max()

    # Sum all rounds to get total rounds played
    return rounds_per_match.sum()


def get_player_kd(player_id: str, team_id: int | None = None):
    df = pd.read_csv(KILL_DATA_CSV)
    # TODO: handle filtering only by games where player was on a particular team
    kills = len(df[df["killer_id"] == player_id])
    deaths = len(df[df["victim_id"] == player_id])
    return kills / deaths


def get_player_kpr(player_id: str, team_id: int | None = None):
    df = pd.read_csv(KILL_DATA_CSV)
    # TODO: handle filtering only by games where player was on a particular team
    kills = len(df[df["killer_id"] == player_id])
    rounds_played = get_player_rounds_played(player_id)
    return kills / rounds_played


def get_player_srv(player_id: str, team_id: int | None = None):
    df = pd.read_csv(KILL_DATA_CSV)
    # TODO: handle filtering only by games where player was on a particular team
    deaths = len(df[df["victim_id"] == player_id])
    rounds_played = get_player_rounds_played(player_id)
    return deaths / rounds_played


def get_player_trade_kill_rate(player_id: str, team_id: int | None = None):
    df = pd.read_csv(KILL_DATA_CSV)
    # TODO: handle filtering only by games where player was on a particular team
    trade_kills = len(df[(df["killer_id"] == player_id) & (df["kill_is_trade"])])
    kills = len(df[df["killer_id"] == player_id])
    return trade_kills / kills


def get_player_traded_death_rate(player_id: str, team_id: int | None = None):
    df = pd.read_csv(KILL_DATA_CSV)
    # TODO: handle filtering only by games where player was on a particular team
    traded_deaths = len(df[(df["victim_id"] == player_id) & (df["death_was_traded"])])
    deaths = len(df[df["victim_id"] == player_id])
    return traded_deaths / deaths


def get_player_obj_rate(player_id: str, team_id: int | None = None):
    # TODO: handle filtering only by games where player was on a particular team
    df = pd.read_csv(OBJ_DATA_CSV)
    objective_plays = len(df[df["player_id"] == player_id])

    rounds_played = get_player_rounds_played(player_id)
    return objective_plays / rounds_played


def get_player_entry_kill_rate(player_id: str, team_id: int | None = None):
    # TODO: handle filtering only by games where player was on a particular team)
    df = pd.read_csv(KILL_DATA_CSV)

    # Get the first kill per round
    first_kills = df.groupby(["match_id", "round_num"]).first().reset_index()

    # Count how many times the given player got the entry kill
    return (first_kills["killer_id"] == player_id).sum() / get_player_rounds_played(
        player_id, team_id
    )


def get_player_entry_death_rate(player_id: str, team_id: int | None = None):
    # TODO: handle filtering only by games where player was on a particular team)
    df = pd.read_csv(KILL_DATA_CSV)

    # Get the first death per round
    first_kills = df.groupby(["match_id", "round_num"]).first().reset_index()

    # Count how many times the given player got the entry kill
    return (first_kills["victim_id"] == player_id).sum() / get_player_rounds_played(
        player_id, team_id
    )


def get_player_headshot_percentage(player_id: str, team_id: int | None = None):
    # TODO: handle filtering only by games where player was on a particular team)
    df = pd.read_csv(KILL_DATA_CSV)

    # Filter kills made by the player
    player_kills = df[df["killer_id"] == player_id]

    # Convert 'headshot' column to boolean if it's stored as a string
    player_kills = player_kills.copy()  # Avoid SettingWithCopyWarning
    player_kills["headshot"] = player_kills["headshot"].astype(bool)

    # Count total kills and headshot kills
    total_kills = len(player_kills)
    headshot_kills = player_kills["headshot"].sum()  # Now correctly counting headshot kills

    # Avoid division by zero
    return (headshot_kills / total_kills * 100) if total_kills > 0 else 0.0


if __name__ == "__main__":
    print(get_player_headshot_percentage("f2d9deea-6c23-4812-a052-eb3bbcc0e1d3"))
