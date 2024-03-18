import os
import shutil
import discord
from dotenv import load_dotenv
import subprocess
import json
import tabulate


cwd = os.getcwd()
os.makedirs('input', exist_ok=True)
os.makedirs('cache', exist_ok=True)
os.makedirs('output', exist_ok=True)
r6_dissect_exe = os.path.join(cwd, 'r6-dissect.exe')

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')

@client.event
async def on_message(message):
    # Check if message has a file
    if not message.attachments:
        return

    # Check if bot is mentioned
    if client.user.mentioned_in(message) == False:
        return

    # Save the file to input/ and unpack to cache/
    await message.attachments[0].save(os.path.join('input', message.attachments[0].filename))
    success = unpack_file(message.attachments[0].filename)
    if success == False:
        await message.reply(f'You sent a file named: {message.attachments[0].filename}. Please send a .zip file.')

    # Parse the replays in cache/ to jsons in output/
    parse_replays()

    # Parse the jsons in output/ and print the results
    outputs = parse_jsons(message)
    print(outputs)
    if type(outputs) == str:
        await message.reply(outputs)
    else:
        for output in outputs:
            await message.reply(output)

    # Delete all files in output/
    for file in os.listdir('output'):
        print(f'Deleting {os.path.join("output", file)}...')
        os.remove(os.path.join(cwd, 'output', file))


def unpack_file(file):
    if not file.endswith('.zip'):
        # Delete original file from input/
        os.remove(os.path.join('input', file))
        return False

    # Unzip the file to cache/
    print(f'Unpacking {os.path.join("input", file)} to cache/...')
    filepath = os.path.join('input', file)
    shutil.unpack_archive(filepath, 'cache')

    # Delete original file from input/
    print(f'Deleting {os.path.join("input", file)}')
    os.remove(filepath)
    return True


def parse_replays():
    # Walk through the cache/ directory and look for Match- folders
    for root, dirs, files in os.walk('cache'):
        for _dir in dirs:
            if _dir.startswith('Match-'):
                print(f'Processing {os.path.join(root, _dir)}...')
                command = f'{r6_dissect_exe} "{os.path.join(cwd, root, _dir)}" -o {os.path.join(cwd, "output", _dir)}.json'
                print(f'Running {command}')
                subprocess.run(command, shell=True, capture_output=True, text=True)

    # Remove all files and directories in cache/
    for root, dirs, files in os.walk('cache'):
        for file in files:
            print(f'Deleting {os.path.join(root, file)}...')
            os.remove(os.path.join(cwd, root, file))

    # TODO: deleting directories doesn't actually work
    for _dir in os.listdir('cache'):
        print(f'Deleting {os.path.join(root)}...')
        os.rmdir(os.path.join(cwd, root))

def parse_jsons(message):
    # sourcery skip: hoist-statement-from-loop, move-assign-in-block
    # Walk through the output/ directory and parse the jsons

    outputs = []
    for file in os.listdir('output'):
        print(f'Parsing {os.path.join("output", file)}...')
        try:
            data = json.load(open(os.path.join('output', file)))

            output_text = ''

            # Figure out which team is us
            teams = data['rounds'][0]['teams']
            uc_names = ['uc', 'cincinnati', 'cinci', 'cincy']
            our_team_idx = -1
            for name in uc_names:
                if name in teams[0]['name'].lower():
                    our_team_idx = 0
                    break
                elif name in teams[1]['name'].lower():
                    our_team_idx = 1
                    break
            if our_team_idx == -1:
                return "Error: couldn't determine which team is UC"
            output_text += f'### {teams[our_team_idx]["name"]} vs {teams[1 - our_team_idx]["name"]}\n'

            # Get date
            output_text += f'Date: {data["rounds"][0]["timestamp"].split("T")[0]}\n'

            # Get map
            output_text += f'Map: {data["rounds"][0]["map"]["name"].lower().capitalize()}\n'

            # Figure out if we're attacking or defending first
            output_text += f'We {teams[our_team_idx]["role"].lower()} first\n'

            # If OT, figure out which side we're on first in OT
            if len(data['rounds']) > 12:
                output_text += f'OT we {data["rounds"][12]["teams"][our_team_idx]["role"].lower()} first\n'

            # Parse result of each round
            # TODO: If all players leave, don't count the round
            round_results = {'attack': {}, 'defense': {}}
            for _round in data['rounds']:

                site = _round['site']
                side = _round['teams'][our_team_idx]['role'].lower()
                win = _round['teams'][our_team_idx]['won']
                if site not in round_results[side]:
                    round_results[side][site] = {'wins': 0, 'losses': 0}
                if win:
                    round_results[side][site]['wins'] += 1
                else:
                    round_results[side][site]['losses'] += 1

            # Calculate overall win-loss round record
            wins, losses = 0, 0
            for key, value in round_results.items():
                for site in value:
                    wins += round_results[key][site]['wins']
                    losses += round_results[key][site]['losses']
            output_text += f'Overall record: {wins}-{losses}\n```\n'

            for key, value_ in round_results.items():
                output_text += f'{key.capitalize()} results:\n'
                for site in sorted(value_, reverse=True):
                    output_text += f'  {site}: {round_results[key][site]["wins"]}-{round_results[key][site]["losses"]}\n'

            headers = ['Player', 'K', 'D', 'A']
            players = []
            kda_table = []
            for i in range(len(data['stats'])):
                player = data['stats'][i]
                for player_info in data['rounds'][0]['players']:
                    if player_info['username'] == player['username']:
                        if player_info['teamIndex'] == our_team_idx:
                            players.append(player)
                        break
            kda_table.extend(
                [
                    player['username'],
                    player['kills'],
                    player['deaths'],
                    player['assists'],
                ]
                for player in players
            )
            output_text += '```\n```\n' + tabulate.tabulate(kda_table, headers=headers) + '```\n'
            output_text += '_Please note that an error exists when tracking round wins for rounds where the defuser is planted. See [r6-dissect Issue 86](https://github.com/redraskal/r6-dissect/issues/86) for more information._\n'
            outputs.append(output_text)
        except json.decoder.JSONDecodeError:
            pass

    return outputs

client.run(TOKEN)