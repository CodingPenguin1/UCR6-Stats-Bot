from ast import literal_eval
from pathlib import Path

import pandas as pd

GAME_INFO_CSV = Path("/home/rjslater/Documents/Projects/UCR6-Stats-Bot/data/3_tables/game_info.csv")
KILL_DATA_CSV = Path("/home/rjslater/Documents/Projects/UCR6-Stats-Bot/data/3_tables/kill_data.csv")
PLAYER_INFO_CSV = Path(
    "/home/rjslater/Documents/Projects/UCR6-Stats-Bot/data/3_tables/player_info.csv"
)
ROUND_DATA_CSV = Path(
    "/home/rjslater/Documents/Projects/UCR6-Stats-Bot/data/3_tables/round_data.csv"
)
TEAM_INFO_CSV = Path("/home/rjslater/Documents/Projects/UCR6-Stats-Bot/data/3_tables/teams.csv")
OP_BAN_CSV = Path("/home/rjslater/Documents/Projects/UCR6-Stats-Bot/data/3_tables/op_bans.csv")


def get_team_id_from_name(team_name: str) -> int | None:
    df = pd.read_csv(TEAM_INFO_CSV)
    df = df[df["team_name"] == team_name]
    if len(df):
        return df.iloc[0]["team_id"]
    return None


def get_game_id_from_match_id(match_id: str) -> int | None:
    df = pd.read_csv(GAME_INFO_CSV)
    for _, row in df.iterrows():
        if match_id in row["match_ids"]:
            return row["game_id"]
    return None


def get_team_id_from_roster(player_ids: list[str]) -> int:
    df = pd.read_csv(TEAM_INFO_CSV)
    # Filter teams down to just those that contain the requested players
    for player_id in player_ids:
        df = df[df["player_ids"].str.contains(player_id)]
    return df.iloc[0]["team_id"]


def get_team_name_from_id(team_id: int) -> str | None:
    df = pd.read_csv(TEAM_INFO_CSV)
    df = df[df["team_id"] == team_id]
    if len(df):
        return df.iloc[0]["team_name"]
    return None


def get_team_ids_from_round_dict(round_dict: dict) -> tuple[int, int]:
    team_0_players = [
        player["profileID"] for player in round_dict["players"] if player["teamIndex"] == 0
    ]
    team_1_players = [
        player["profileID"] for player in round_dict["players"] if player["teamIndex"] == 1
    ]
    team_0_id = get_team_id_from_roster(team_0_players)
    team_1_id = get_team_id_from_roster(team_1_players)
    return team_0_id, team_1_id


def get_players_from_team_name(team_name: str) -> list[str] | None:
    teams = pd.read_csv(TEAM_INFO_CSV)
    teams = teams[teams["team_name"] == team_name]
    if len(teams):
        return get_player_ids_from_team_id(teams.iloc[0]["team_id"])
    return None


def get_player_ids_from_team_id(team_id: int) -> list[str] | None:
    teams = pd.read_csv(TEAM_INFO_CSV)
    teams = teams[teams["team_id"] == team_id]
    if len(teams):
        return literal_eval(teams.iloc[0]["player_ids"])
    return None


def get_player_id_from_name(player_name: str) -> str | None:
    df = pd.read_csv(PLAYER_INFO_CSV)
    df = df[df["player_names"].str.contains(f"'{player_name}'")]
    if len(df):
        return df.iloc[0]["player_id"]
    return None


def get_player_names_from_id(player_id: str) -> list[str] | None:
    df = pd.read_csv(PLAYER_INFO_CSV)
    df = df[df["player_id"] == player_id]
    if len(df):
        return literal_eval(df.iloc[0]["player_names"])
    return None


def get_player_name_from_id(player_id: str) -> str | None:
    return get_player_names_from_id(player_id)[0]


def player_is_on_team(player_id: str, team_id: int) -> bool:
    teams = pd.read_csv(TEAM_INFO_CSV)
    team = teams[teams["team_id"] == team_id].iloc[0]
    return player_id in literal_eval(team["player_ids"])


if __name__ == "__main__":
    ...
    ...
