
config = [
	{
		'name': "Accuracy/Accuracy",
		'args': [
			"--log-dir", "logs/accuracy",
			"--workers", "8",
			"--epochs", "10",
			"--batch-size", "30",
			"--hidden-layers", "128",
			"--runs", "10",
			"--test-set"
		],
		'subtests': [
			{
				'name': "CPU",
				'executable': "cpu/digits_demo.exe"
			},
			{
				'name': "CUDA/Float",
				'executable': "cuda_float/digits_demo.exe"
			},	
			{
				'name': "CUDA/Mixed",
				'executable': "cuda_half/digits_demo.exe"
			},
			{
				'name': "CUDA/WMMA",
				'executable': "cuda_wmma/digits_demo.exe"
			}
		]
	},
	{
		'name': "Accuracy/Performance",
		'args': [
			"--log-dir", "logs/accuracy",
			"--workers", "8",
			"--epochs", "10",
			"--batch-size", "30",
			"--hidden-layers", "128",
			"--runs", "10"
		],
		'subtests': [
			{
				'name': "CPU",
				'executable': "cpu/digits_demo.exe"
			},
			{
				'name': "CUDA/Float",
				'executable': "cuda_float/digits_demo.exe"
			},	
			{
				'name': "CUDA/Mixed",
				'executable': "cuda_half/digits_demo.exe"
			},
			{
				'name': "CUDA/WMMA",
				'executable': "cuda_wmma/digits_demo.exe"
			}
		]
	},
	{
		'name': "Performance/Scaling",
		'args': [
			"--log-dir", "logs/scaling",
			"--workers", "20",
			"--epochs", "1",
			"--batch-size", "32",
			"--hidden-layers", "4,6,8,10,12,14,16,18,20,28,32,64,128,256,512,768,1024,2048",
			"--runs", "10"
		],
		'subtests': [
			{
				'name': "CPU",
				'executable': "cpu/digits_demo.exe"
			},
			{
				'name': "CUDA/Float",
				'executable': "cuda_float/digits_demo.exe"
			},	
			{
				'name': "CUDA/Mixed",
				'executable': "cuda_half/digits_demo.exe"
			},
			{
				'name': "CUDA/WMMA",
				'executable': "cuda_wmma/digits_demo.exe"
			}
		]
	},
	{
		'name': "Performance/Threading",
		'args': [
			"--log-dir", "logs/threading",
			"--workers", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32",
			"--epochs", "1",
			"--batch-size", "32",
			"--hidden-layers", "512",
			"--runs", "10"
		],
		'subtests': [
			{
				'name': "CPU",
				'executable': "cpu/digits_demo.exe"
			}
		]
	},
	{
		'name': "Power",
		'args': [
			"--log-dir", "logs/power",
			"--workers", "28",
			"--epochs", "1",
			"--batch-size", "32",
			"--hidden-layers", "32,128,512,1024,4096",
			"--runs", "10",
			"--power",
			"--power-src", "C:\\Users\\chrum\\Downloads\\benchmark\\power.csv"
		],
		'subtests': [
			{
				'name': "CPU",
				'executable': "cpu/digits_demo.exe"
			},
			{
				'name': "CUDA/Float",
				'executable': "cuda_float/digits_demo.exe"
			},	
			{
				'name': "CUDA/Mixed",
				'executable': "cuda_half/digits_demo.exe"
			},
			{
				'name': "CUDA/WMMA",
				'executable': "cuda_wmma/digits_demo.exe"
			}
		]
	},
]

import subprocess
import time

if __name__ == "__main__":

	total_tests = sum(len(group['subtests']) for group in config)
	i = 1
	for group in config:
		for test in group['subtests']:
			print(f"[{i} / {total_tests}] {group['name']}/{test['name']}")
			i += 1
			cmd = ['exec/' + test['executable']] + group['args']
			print(f"        Command: {' '.join(cmd)}")
			start_time = time.time()
			result = subprocess.run(cmd, capture_output=False)
			end_time = time.time()
			if result.returncode != 0:
				print(f"    Test failed")
			else:
				print(f"    Test completed successfully in {end_time - start_time:.2f} seconds")
			time.sleep(1)

