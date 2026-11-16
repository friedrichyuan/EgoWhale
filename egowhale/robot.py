"""Compatibility names for old ARX callers. Nodes use egowhale.robots explicitly."""
import os
import tempfile
from functools import cache
from pathlib import Path

import yaml

from egowhale.robots.arx import ARM as ARM
from egowhale.robots.arx import GRIPPERS as GRIPPERS
from egowhale.robots.arx import SCENE as SCENE
from egowhale.robots.arx import URDF as URDF
from egowhale.robots.arx import YAML as YAML
from egowhale.robots.arx import ARXRobot

TCP = ARXRobot.tcp_sites
CAMERA = ARXRobot.camera


@cache
def robot_yaml():
    path = Path(tempfile.gettempdir()) / f"egowhale_arx_{os.getpid()}.yml"
    path.write_text(yaml.safe_dump(ARXRobot().groups[0].config))
    return str(path)


def hide(model):
    ARXRobot().hide(model)


def place_base(model, transform):
    ARXRobot().place_bases(model, {"base": transform})


def place_camera(model, intrinsic, height):
    ARXRobot().place_camera(model, intrinsic, height)
