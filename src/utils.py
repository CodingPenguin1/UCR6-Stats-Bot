import pandas as pd
from pathlib import Path
from ast import literal_eval

GAME_INFO_CSV = Path("/home/rjslater/Documents/Projects/UCR6-Stats-Bot/data/3_tables/game_info.csv")
KILL_DATA_CSV = Path("/home/rjslater/Documents/Projects/UCR6-Stats-Bot/data/3_tables/kill_data.csv")
PLAYER_INFO_CSV = Path("/home/rjslater/Documents/Projects/UCR6-Stats-Bot/data/3_tables/player_info.csv")
ROUND_DATA_CSV = Path("/home/rjslater/Documents/Projects/UCR6-Stats-Bot/data/3_tables/round_data.csv")
TEAM_INFO_CSV = Path("/home/rjslater/Documents/Projects/UCR6-Stats-Bot/data/3_tables/teams.csv")


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
    team_0_players = [player["profileID"] for player in round_dict["players"] if player["teamIndex"] == 0]
    team_1_players = [player["profileID"] for player in round_dict["players"] if player["teamIndex"] == 1]
    team_0_id = get_team_id_from_roster(team_0_players)
    team_1_id = get_team_id_from_roster(team_1_players)
    return team_0_id, team_1_id


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
    print(player_is_on_team("2261f354-f845-4dbb-8fc9-5f9701a461cd", 1))

    # player_id = get_player_id_from_name("Chalk.UC")
    # print(player_id)
    # print(get_player_name_from_id(player_id))

    # player_ids = [
    #     "ff6ffd37-2c4f-4b86-8de6-9d4b5dfa0eb8",
    #     "4bf5246c-2cad-4023-aefd-71ad4e9e7148",
    #     "b0e6b1ef-2c4a-4d0b-9f46-1e17aa9a80f7",
    #     "0daab651-3299-4469-923b-95817d877cde",
    #     "2261f354-f845-4dbb-8fc9-5f9701a461cd",
    # ]

    # player_ids = [
    #     "ceec0dfe-2e55-4eb4-8a5d-b7df3dd20460",
    #     "f1256865-ad88-41a2-ba5f-18aeae19ff10",
    #     "37cbfe44-4d71-4cee-8c64-d7dc0e8f1dde",
    #     "7186a481-79fc-4a1a-a685-d0cb80af43d4",
    #     "7e5c9b59-2745-4126-bca4-3394eadaa7de",
    # ]

    # print(get_team_id_from_roster(player_ids))
