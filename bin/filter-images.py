#!/usr/bin/env -S poetry run python3

from json import loads
from pathlib import Path

import click
from tqdm import tqdm


@click.command()
@click.option("--source", required=True, type=Path)
@click.option("--target", required=True, type=Path)
def cli(source: Path, target: Path):
    available = set()

    with target.open("a+") as list:
        for file in tqdm(source.glob("*"), desc="Loading metadata"):
            with file.open("r") as lines:
                for line in lines:
                    current = loads(line)

                    name = current["id"]
                    bucket = current["id"][-4:]
                    extension = current["file_ext"]

                    if (
                        current["rating"] == "s"
                        and extension in {"png", "jpg"}
                        and 96 <= int(current["image_width"]) <= 960
                        and 96 <= int(current["image_height"]) <= 960
                    ):
                        print(f"{bucket}/{name}.{extension}", file=list)


if __name__ == "__main__":
    cli()
