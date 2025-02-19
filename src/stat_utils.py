import pandas as pd

from utils import KILL_DATA_CSV, OP_BAN_CSV, ROUND_DATA_CSV, player_is_on_team


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
    rounds_played = len(df["round_id"].unique())
    return kills / rounds_played


def get_player_srv(player_id: str, team_id: int | None = None):
    df = pd.read_csv(KILL_DATA_CSV)
    # TODO: handle filtering only by games where player was on a particular team
    deaths = len(df[df["victim_id"] == player_id])
    rounds_played = len(df["round_id"].unique())
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


if __name__ == "__main__":
    get_team_opening_kill_rate(1)
