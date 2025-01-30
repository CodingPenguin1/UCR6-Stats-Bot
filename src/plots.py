from utils import player_is_on_team, get_player_name_from_id, get_team_name_from_id, KILL_DATA_CSV, ROUND_DATA_CSV
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker


def plot_kds_over_match(match_id: str, team_id: int) -> plt.figure:
    kills = pd.read_csv(KILL_DATA_CSV)
    kills = kills[kills["match_id"] == match_id]
    round_count = max(kills["round_num"])

    kill_counts = {}
    death_counts = {}
    player_ids = [pid for pid in set(kills["killer_id"]) & set(kills["victim_id"]) if player_is_on_team(pid, team_id)]
    for player_id in player_ids:
        if player_is_on_team(player_id, team_id):
            kill_counts[player_id] = []
            death_counts[player_id] = []

    for round_num in range(1, round_count + 1):
        for player_id in player_ids:
            kill_counts[player_id].append(len(kills[(kills["round_num"] == round_num) & (kills["killer_id"] == player_id)]))
            death_counts[player_id].append(len(kills[(kills["round_num"] == round_num) & (kills["victim_id"] == player_id)]))

    kds = {pid: [] for pid in kill_counts}
    for player_id in player_ids:
        for i in range(len(kill_counts[player_id])):
            kill_count = sum(kill_counts[player_id][: i + 1])
            death_count = sum(death_counts[player_id][: i + 1])
            if death_count > 0:
                kd = kill_count / death_count
            else:
                kd = kill_count
            kds[player_id].append(kd)

    rounds = pd.read_csv(ROUND_DATA_CSV)
    rounds = rounds[rounds["match_id"] == match_id]

    xticks = []
    for round_num in range(1, round_count + 1):
        kill_data = kills[kills["round_num"] == round_num].iloc[0]
        if player_is_on_team(kill_data["killer_id"], team_id):
            side = "ATK" if kill_data["killer_side"] == "Attack" else "DEF"
        else:
            side = "ATK" if kill_data["victim_side"] == "Attack" else "DEF"

        round_data = rounds[rounds["round_num"] == round_num].iloc[0]
        site = round_data["site"]
        xtick = f"R{round_num} {side} {site.split(' ')[1]}".removesuffix(",")
        xticks.append(xtick)
        print(xtick)

    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_players = sorted(kds.keys(), key=lambda pid: get_player_name_from_id(pid).lower())
    for player_id in sorted_players:
        ax.plot(kds[player_id], label=get_player_name_from_id(player_id))

    map_name = kills[kills["match_id"] == match_id].iloc[0]["map"]
    ax.set_title(map_name)
    ax.set_xlabel("Round")
    ax.set_ylabel("K/D")
    ax.set_xticks(ticks=range(len(xticks)), labels=xticks, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    return fig


def plot_kills_deaths_time_boxplots(player_id: str) -> plt.figure:
    """
    Plots a dot scatter plot with an overlaid box-and-whisker plot for each category.

    Args:
        data_dict (dict[str, list[int]]): Dictionary where keys are categories and values are lists of time in seconds.
        title (str): Title for the plot.
    """

    def seconds_to_mmss(seconds):
        """Convert seconds to MM:SS format."""
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02}:{seconds:02}"

    def plot_scatter_with_boxplot(ax, data_dict, title=""):
        categories = list(data_dict.keys())
        if not categories:  # If there's no data to plot, skip
            return

        x_positions = range(1, len(categories) + 1)  # Boxplot expects positions starting from 1

        # Use a consistent color palette for categories
        palette = sns.color_palette("husl", len(categories))

        # Overlay scatter plot and box plot
        for x, (category, color) in zip(x_positions, zip(categories, palette)):
            y_values = data_dict[category]

            # Skip empty lists
            if not y_values:
                continue

            # Scatter plot points
            ax.scatter([x] * len(y_values), y_values, color=color, alpha=0.7)

            # Box plot for each category
            ax.boxplot(
                [y_values],
                positions=[x],
                widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.5),
                medianprops=dict(color="black"),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
                flierprops=dict(markeredgecolor=color),
            )

        # Customize x-axis and y-axis
        ax.set_xticks(x_positions)
        ax.set_xticklabels(categories, rotation=45)
        ax.set_xlabel("Map")
        ax.set_ylabel("Time")
        ax.set_ylim(0, 3 * 60)

        # Apply the title if provided
        if title:
            ax.set_title(title)

        # Format y-axis labels as MM:SS
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: seconds_to_mmss(int(x))))

        ax.grid(axis="y", linestyle="--", alpha=0.7)

    kill_df = pd.read_csv(KILL_DATA_CSV)
    kill_times_atk = {}  # {map: [kill_times]}
    kill_times_def = {}  # {map: [kill_times]}
    death_times_atk = {}  # {map: [death_times]}
    death_times_def = {}  # {map: [death_times]}

    for map_name in kill_df["map"].unique():
        kill_times_atk[map_name] = list(
            kill_df[(kill_df["map"] == map_name) & (kill_df["killer_side"] == "Attack") & (kill_df["killer_id"] == player_id)]["time_seconds"]
        )
        death_times_atk[map_name] = list(
            kill_df[(kill_df["map"] == map_name) & (kill_df["victim_side"] == "Attack") & (kill_df["victim_id"] == player_id)]["time_seconds"]
        )
        kill_times_def[map_name] = list(
            kill_df[(kill_df["map"] == map_name) & (kill_df["killer_side"] == "Defense") & (kill_df["killer_id"] == player_id)]["time_seconds"]
        )
        death_times_def[map_name] = list(
            kill_df[(kill_df["map"] == map_name) & (kill_df["victim_side"] == "Defense") & (kill_df["victim_id"] == player_id)]["time_seconds"]
        )

    data_dict_list = [
        kill_times_atk,
        kill_times_def,
        death_times_atk,
        death_times_def,
    ]
    titles_list = [
        "Times of Kills (ATK)",
        "Times of Kills (DEF)",
        "Times of Deaths (ATK)",
        "Times of Deaths (DEF)",
    ]

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Flatten the 2x2 axs array to make it easier to iterate over
    axs = axs.flatten()

    for i, (data_dict, title) in enumerate(zip(data_dict_list, titles_list)):
        # Plot each of the 4 boxplots on a different subplot
        plt.sca(axs[i])  # Set current axes to the ith subplot
        plot_scatter_with_boxplot(axs[i], data_dict, title=title)

    # Set title and such
    plt.suptitle(f"{get_player_name_from_id(player_id)} Kill/Death Times")
    plt.tight_layout()
    return fig


# def plot_engagement_efficiency(player_id: str) -> plt.figure:
#     kill_df = pd.read_csv(KILL_DATA_CSV)
#     kill_df[(kill_df["killer_id"] == player_id) & (kill_df["map"] == "Skyscraper") & (kill_df["time_seconds"] > 140)]


if __name__ == "__main__":
    # match_id = "Match-2025-01-28_20-06-55-36144"
    # match_id = "Match-2025-01-28_21-27-26-36144"
    # team_id = 0
    # fig = plot_kds_over_match(match_id, team_id)
    # fig.savefig(
    #     Path("/home/rjslater/Documents/Projects/UCR6-Stats-Bot/plots") / f"KD_{get_team_name_from_id(team_id)}_{match_id}", bbox_inches="tight"
    # )

    player_id = "b0e6b1ef-2c4a-4d0b-9f46-1e17aa9a80f7"
    fig = plot_kills_deaths_time_boxplots(player_id)
    fig.savefig(Path("/home/rjslater/Documents/Projects/UCR6-Stats-Bot/plots") / "kills_deaths_boxplots")
