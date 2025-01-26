import os
import discord
from dotenv import load_dotenv
from discord import app_commands
from json import load as json_load


load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)


@client.event
async def on_ready():
    await tree.sync()
    print(f"{client.user} has connected to Discord!")


@tree.command(name="hello")
async def hello(interaction: discord.Interaction):
    await interaction.response.send_message("Hello World!")


if __name__ == "__main__":
    with open("config.json") as f:
        CONFIG = json_load(f)
    client.run(TOKEN)
