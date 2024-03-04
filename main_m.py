
from utils.config import process_config, setup_logger, process_config_default
from agents.general_agent import *

import argparse

def main(config_path, default_config_path):
    setup_logger()

    config = process_config_default(config_path, default_config_path)
    agent_class = globals()[config.agent]
    agent = agent_class(config)
    agent.run()
    agent.finalize()


parser = argparse.ArgumentParser(description="My Command Line Program")
parser.add_argument('--config', help="Number of config file")
parser.add_argument('--default_config', help="Number of config file")
args = parser.parse_args()
print(args)


main(config_path=args.config, default_config_path=args.default_config)