import sys
sys.path.append('Lib/')

import argparse
import graph_utils

from simile_predict import *

def test_simile(configfile):

	simile = SIMILE_PREDICT (configfile)
	rollout = simile.policy_rollout()
	simile.graph.plot_rollout_test(rollout)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-config', default='config_test.ini', help='Config file with parameters for testing')
	args = parser.parse_args()

	if not os.path.isfile(args.config):
		print ('Error: configfile', args.config, 'does not exist.')
		exit()
	
	test_simile(args.config)


