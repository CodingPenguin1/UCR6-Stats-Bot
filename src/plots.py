from pathlib import Path
from typing import Iterable

import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from stat_utils import (
    get_player_kd,
    get_player_kpr,
    get_player_srv,
    get_player_trade_kill_rate,
    get_player_traded_death_rate,
    get_team_opening_kill_rate,
    get_team_winrate,
)
from utils import (
    KILL_DATA_CSV,
    ROUND_DATA_CSV,
    get_player_ids_from_team_id,
    get_player_name_from_id,
    get_team_id_from_name,
    get_team_name_from_id,
    player_is_on_team,
)

PLOTS_DIR = Path("/home/rjslater/Documents/Projects/UCR6-Stats-Bot/plots")
PLOTS_DIR = Path(r"C:\Users\ryanj\Documents\UCR6-Stats-Bot\plots")


def plot_kds_over_match(match_id: str, team_id: int) -> plt.figure:
    kills = pd.read_csv(KILL_DATA_CSV)
    kills = kills[kills["match_id"] == match_id]
    round_count = max(kills["round_num"])

    kill_counts = {}
    death_counts = {}
    player_ids = [
        pid
        for pid in set(kills["killer_id"]) & set(kills["victim_id"])
        if player_is_on_team(pid, team_id)
    ]
    for player_id in player_ids:
        if player_is_on_team(player_id, team_id):
            kill_counts[player_id] = []
            death_counts[player_id] = []

    for round_num in range(1, round_count + 1):
        for player_id in player_ids:
            kill_counts[player_id].append(
                len(kills[(kills["round_num"] == round_num) & (kills["killer_id"] == player_id)])
            )
            death_counts[player_id].append(
                len(kills[(kills["round_num"] == round_num) & (kills["victim_id"] == player_id)])
            )

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


def plot_scatter_with_boxplot(ax, data_dict, title="", x_label: str = "Map", y_label: str = "Time"):
    def seconds_to_mmss(seconds):
        """Convert seconds to MM:SS format."""
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02}:{seconds:02}"

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
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(0, 3 * 60)

    # Apply the title if provided
    if title:
        ax.set_title(title)

    # Format y-axis labels as MM:SS
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: seconds_to_mmss(int(x))))

    ax.grid(axis="y", linestyle="--", alpha=0.7)


def plot_player_engagement_timing(
    player_id: str, match_ids: None | Iterable[str] = None
) -> plt.figure:
    """
    Plots a dot scatter plot with an overlaid box-and-whisker plot for each category.

    Args:
        data_dict (dict[str, list[int]]): Dictionary where keys are categories and values are lists of time in seconds.
        title (str): Title for the plot.
    """

    kill_df = pd.read_csv(KILL_DATA_CSV)
    # Filter by match IDs if specified
    if match_ids is not None and len(match_ids):
        kill_df = kill_df[kill_df["match_id"].isin(match_ids)]

    kill_times_atk = {}  # {map: [kill_times]}
    kill_times_def = {}  # {map: [kill_times]}
    death_times_atk = {}  # {map: [death_times]}
    death_times_def = {}  # {map: [death_times]}

    for map_name in sorted(kill_df["map"].unique()):
        kill_times_atk[map_name] = list(
            kill_df[
                (kill_df["map"] == map_name)
                & (kill_df["killer_side"] == "Attack")
                & (kill_df["killer_id"] == player_id)
            ]["time_seconds"]
        )
        death_times_atk[map_name] = list(
            kill_df[
                (kill_df["map"] == map_name)
                & (kill_df["victim_side"] == "Attack")
                & (kill_df["victim_id"] == player_id)
            ]["time_seconds"]
        )
        kill_times_def[map_name] = list(
            kill_df[
                (kill_df["map"] == map_name)
                & (kill_df["killer_side"] == "Defense")
                & (kill_df["killer_id"] == player_id)
            ]["time_seconds"]
        )
        death_times_def[map_name] = list(
            kill_df[
                (kill_df["map"] == map_name)
                & (kill_df["victim_side"] == "Defense")
                & (kill_df["victim_id"] == player_id)
            ]["time_seconds"]
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


def get_engagement_data(
    player_id: str,
    split_atk_def: bool = False,
    match_ids: None | Iterable[str] = None,
    time_bin_width: int = 15,
):
    kill_df = pd.read_csv(KILL_DATA_CSV)
    kill_df = kill_df[(kill_df["killer_id"] == player_id) | (kill_df["victim_id"] == player_id)]
    if match_ids is not None and len(match_ids):
        kill_df = kill_df[kill_df["match_id"].isin(match_ids)]

    if split_atk_def:
        maps = []
        for map_name in sorted(kill_df["map"].unique()):
            maps.extend([f"{map_name} (ATK)", f"{map_name} (DEF)"])
    else:
        maps = sorted(kill_df["map"].unique())
    time_bins = list(range(60 * 3, 0, -time_bin_width))  # Upper bounds of bins

    kills = np.zeros((len(maps), len(time_bins)), dtype=np.uint)
    deaths = np.zeros((len(maps), len(time_bins)), dtype=np.uint)

    for _, row in kill_df.iterrows():
        _map = row["map"]
        kill = row["killer_id"] == player_id
        time = row["time_seconds"]

        if split_atk_def:
            for row_idx, map_side in enumerate(maps):
                map_name = map_side.split(" ")[0]
                side = "Attack" if "ATK" in map_side else "Defense"
                player_side = (
                    row["killer_side"] if row["killer_id"] == player_id else row["victim_side"]
                )
                if row["map"] == map_name and player_side == side:
                    break

        else:
            row_idx = maps.index(_map)

        col_idx = len(time_bins) - 1
        for i, time_bin_upper_bound in enumerate(time_bins):
            if time > time_bin_upper_bound:
                col_idx = i - 1
                break

        if kill:
            kills[row_idx][col_idx] += 1
        else:
            deaths[row_idx][col_idx] += 1

    engagement_counts = kills + deaths
    engagement_winrate = np.minimum(
        np.divide(
            kills,
            kills + deaths,
            where=(kills + deaths) != 0,
            out=np.full_like(kills, fill_value=0, dtype=float),
        ),
        1,
    )

    # If no data, set winrate to -1
    engagement_winrate[(kills + deaths) == 0] = -1

    return engagement_counts, engagement_winrate, maps, time_bins


def get_traded_death_engagement_data(
    player_id: str,
    split_atk_def: bool = False,
    match_ids: None | Iterable[str] = None,
    time_bin_width: int = 15,
):
    kill_df = pd.read_csv(KILL_DATA_CSV)
    kill_df = kill_df[(kill_df["killer_id"] == player_id) | (kill_df["victim_id"] == player_id)]
    if match_ids is not None and len(match_ids):
        kill_df = kill_df[kill_df["match_id"].isin(match_ids)]

    # Filter to player id
    kill_df = kill_df[kill_df["victim_id"] == player_id]

    if split_atk_def:
        maps = []
        for map_name in sorted(kill_df["map"].unique()):
            maps.extend([f"{map_name} (ATK)", f"{map_name} (DEF)"])
    else:
        maps = sorted(kill_df["map"].unique())
    time_bins = list(range(60 * 3, 0, -time_bin_width))  # Upper bounds of bins

    deaths = np.zeros((len(maps), len(time_bins)), dtype=np.uint)
    traded_deaths = np.zeros((len(maps), len(time_bins)), dtype=np.uint)

    for row_idx, map_str in enumerate(maps):
        for col_idx, time_bin in enumerate(time_bins):
            map_name = map_str if "(" not in map_str else map_str.split(" ")[0]

            # Apply map and time bin
            binned_deaths_df = kill_df[
                (kill_df["map"] == map_name)
                & (kill_df["time_seconds"] < time_bin)
                & (kill_df["time_seconds"] > time_bin - time_bin_width)
            ]

            if split_atk_def:
                side = "Attack" if "(A" in map_str else "Defense"
                binned_deaths_df = binned_deaths_df[binned_deaths_df["victim_side"] == side]

            # Find traded deaths
            deaths[row_idx][col_idx] = len(binned_deaths_df)
            traded_deaths[row_idx][col_idx] = len(
                binned_deaths_df[binned_deaths_df["death_was_traded"]]
            )

    # Calculate the traded death ratio
    traded_death_rate = np.divide(
        traded_deaths,
        deaths,
        where=deaths != 0,
        out=np.full_like(traded_deaths, fill_value=0, dtype=float),
    )

    # If no data, set traded death rate to -1
    traded_death_rate[deaths == 0] = -1

    return deaths, traded_death_rate, maps, time_bins


def plot_engagement_efficiency(
    engagement_counts: np.array,
    engagement_winrate: np.array,
    maps: list[str],
    time_bins: list[int],
    title: str = "Engagement Efficiency",
    legend_text: str = "Color Legend:\nBlue = High KpE\nGray = Mid KpE\nRed = Low KpE\n\nCircles: # Engagements",
) -> plt.figure:
    def sec_to_mm_ss(seconds: int) -> str:
        """Convert seconds to MM:SS format."""
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:2}:{seconds:02}"

    fig, ax = plt.subplots(figsize=(10, 6))

    # Format time bins using sec_to_mm_ss
    formatted_time_bins = [
        f"{sec_to_mm_ss(start)} - {sec_to_mm_ss(start + (time_bins[1] - time_bins[0]))}"
        for start in time_bins
    ]

    # Create mask for cells where engagement_counts == 0
    mask = engagement_counts == 0

    # Define custom colormap: Bad/Low Value (#ae282c), Mid (#ededed), Good/High Value (#2066a8)
    colors = ["#ae282c", "#ededed", "#2066a8"]  # Red (Low), Grey (Mid), Blue (High)
    n_bins = 100  # Number of bins
    cmap_name = "custom_cmap"
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # Plot heatmap without a colorbar
    sns.heatmap(
        engagement_winrate * 100,
        annot=False,
        cmap=cmap,
        vmin=0,
        vmax=100,
        xticklabels=formatted_time_bins,
        yticklabels=maps,
        mask=mask,  # Hide cells where engagement_counts == 0
        ax=ax,
        cbar=True,  # Remove colorbar
    )

    # Overlay gray bullets sized proportionally to engagement counts (only for non-zero values)
    y_coords, x_coords = np.meshgrid(np.arange(len(maps)), np.arange(len(time_bins)), indexing="ij")
    valid_mask = engagement_counts > 0  # Only show bullets for nonzero engagement counts
    sizes = (engagement_counts[valid_mask] / np.max(engagement_counts[valid_mask]) * 500).flatten()

    scatter = ax.scatter(
        x_coords[valid_mask] + 0.5, y_coords[valid_mask] + 0.5, s=sizes, color="white", alpha=0.6
    )

    # Add values on top of bullets (only for nonzero engagement counts)
    for i in range(len(y_coords.flatten())):
        if engagement_counts.flatten()[i] > 0:
            ax.text(
                x_coords.flatten()[i] + 0.5,
                y_coords.flatten()[i] + 0.5,
                f"{int(engagement_counts.flatten()[i])}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                fontweight="bold",
            )

    ax.set_xlabel("Time")
    ax.set_ylabel("Map")
    ax.set_title(title)  # Use the title parameter

    # Rotate x-ticks 45 degrees counterclockwise
    plt.xticks(rotation=45)

    # Add text annotation explaining colors
    text_x = -3.5  # Position slightly outside the heatmap
    text_y = len(maps)  # Position below the last row
    ax.text(
        text_x,
        text_y,
        legend_text,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
    )

    plt.tight_layout()
    return fig


def plot_team_engagement_timings(team_id: int, map_name: str | None = None) -> plt.figure:
    player_ids = get_player_ids_from_team_id(team_id)
    player_names = [get_player_name_from_id(player_id) for player_id in player_ids]
    kill_df = pd.read_csv(KILL_DATA_CSV)
    if map_name is not None:
        kill_df = kill_df[kill_df["map"] == map_name]

    kill_times = {player_name: [] for player_name in player_names}
    death_times = {player_name: [] for player_name in player_names}

    for player_id, player_name in zip(player_ids, player_names):
        kill_times[player_name] = list(
            kill_df[kill_df["killer_id"] == player_id]["time_seconds"].values
        )
        death_times[player_name] = list(
            kill_df[kill_df["victim_id"] == player_id]["time_seconds"].values
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    plot_scatter_with_boxplot(
        axes[0], kill_times, title="Kill Timings", x_label="Player", y_label="Time"
    )
    plot_scatter_with_boxplot(
        axes[1], death_times, title="Death Timings", x_label="Player", y_label="Time"
    )

    plt.tight_layout()
    return fig


def radar_plot(categories: list[str], values: dict[str : list[int]]):
    def add_ring(ax, ring_values, color="blue"):
        angles = np.linspace(0, 2 * np.pi, len(ring_values), endpoint=False).tolist()
        ring_values += ring_values[:1]
        angles += angles[:1]
        ax.plot(angles, ring_values, color=color, linewidth=2, linestyle="solid")
        # ax.plot(angles, ring_values, color=color, linewidth=2, linestyle="solid", marker="o")

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    colors = sns.color_palette()
    for color, (value_label, value_list) in zip(colors, values.items()):
        add_ring(ax, value_list, color)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    plt.xticks(angles, categories)

    # Remove y ticks down to max needed to show all data
    max_value = max(max(value_list) for value_list in values.values())
    ticks = np.arange(0.25, max_value + 0.25, 0.25)

    plt.yticks(
        ticks,
        [f"{t:>1}" for t in ticks],
        color="grey",
        size=8,
    )
    plt.ylim(0, ticks[-1])
    if len(values.keys()):
        plt.legend(values.keys())
    else:
        plt.title(list(values.keys())[0])

    plt.show()


if __name__ == "__main__":
    # team_id = get_team_id_from_name("UC Black")
    team_id = get_team_id_from_name("CINCINNATI")

    radar_plot_data = {}
    for player_id in get_player_ids_from_team_id(team_id):
        player_name = get_player_name_from_id(player_id)

        # Player radar plot
        print()
        print(player_name)
        kd = get_player_kd(player_id)
        kpr = get_player_kpr(player_id)
        srv = get_player_srv(player_id)
        trade_kill_rate = get_player_trade_kill_rate(player_id)
        traded_death_rate = get_player_traded_death_rate(player_id)
        # TODO: add plant rate
        print(f"{kd=}")
        print(f"{kpr=}")
        print(f"{srv=}")
        print(f"{trade_kill_rate=}")
        print(f"{traded_death_rate=}")
        # radar_plot_data[player_name] = [kd, kpr, srv, trade_kill_rate, traded_death_rate]
        radar_plot_data[player_name] = [kpr, srv, trade_kill_rate, traded_death_rate]

        # fig = plot_player_engagement_timing(player_id)
        # fig.savefig(PLOTS_DIR / f"engagement_timing-{player_name}.png")

        # death_counts, traded_death_rate, maps, time_bins = get_traded_death_engagement_data(
        #     player_id, split_atk_def=True
        # )
        # plot_engagement_efficiency(
        #     death_counts,
        #     traded_death_rate,
        #     maps,
        #     time_bins,
        #     title=f"Traded Death Rate - {player_name}",
        #     legend_text="Color Legend:\nBlue = High trade rate\nGray = Mid trade rate\nRed = Low trade rate\n\nCircles: # deaths",
        # ).savefig(PLOTS_DIR / f"traded_death_rates-{player_name}.png")

        # engagement_counts, engagement_winrate, maps, time_bins = get_engagement_data(
        #     player_id, split_atk_def=True
        # )
        # fig = plot_engagement_efficiency(
        #     engagement_counts,
        #     engagement_winrate,
        #     maps,
        #     time_bins,
        #     title=f"Engagement Efficiency - {player_name}",
        # )
        # fig.savefig(PLOTS_DIR / f"engagement_efficiency-{player_name}.png")

    # team_name = get_team_name_from_id(team_id)
    # plot_team_engagement_timings(team_id).savefig(PLOTS_DIR / f"engagement_timing-{team_name}.png")

    # radar_plot(["K/D", "KPR", "SRV", "Trade Kill Rate", "Trade Death Rate"], radar_plot_data)
    radar_plot(["KPR", "SRV", "Trade Kill Rate", "Trade Death Rate"], radar_plot_data)

    team_ids = [get_team_id_from_name("UC Black"), get_team_id_from_name("CINCINNATI")]
    stats = {}
    for team_id in team_ids:
        team_name = get_team_name_from_id(team_id)
        atk_winrate = get_team_winrate(team_id, side="Attack")[0]
        def_winrate = get_team_winrate(team_id, side="Defense")[0]
        opening_kill_rate = get_team_opening_kill_rate(team_id)[0]
        stats[team_name] = [atk_winrate, def_winrate, opening_kill_rate]
        print(team_name)
        print(f"{atk_winrate=}")
        print(f"{def_winrate=}")
        print(f"{opening_kill_rate=}")

    radar_plot(
        ["ATK Win Rate %", "DEF Win Rate %", "Opening Kill %"],
        stats,
    )  #     print(f"{opening_kill_rate=}")
