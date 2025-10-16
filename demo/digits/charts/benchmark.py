
config = [
	{
		'name': "Accuracy/Accuracy",
		'args': [
			"--log-dir", "logs/accuracy",
			"--workers", "4",
			"--epochs", "10",
			"--batch-size", "30",
			"--hidden-layers", "128",
			"--iterations", "10",
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
				'executable': "cuda_mixed/digits_demo.exe"
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
			"--workers", "4",
			"--epochs", "10",
			"--batch-size", "30",
			"--hidden-layers", "128",
			"--iterations", "10"
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
			"--workers", "6",
			"--epochs", "10",
			"--batch-size", "30",
			"--hidden-layers", "4,6,8,10,12,14,16,18,20,28,32,64,128,256,512,768,1024",
			"--iterations", "10"
		],
		'subtests': [
			{
				'name': "CPU",
				'executable': "cpu_digits_demo.exe"
			},
			{
				'name': "CUDA/Float",
				'executable': "cuda_float/digits_demo.exe"
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
			"--workers", "6",
			"--epochs", "10",
			"--batch-size", "30",
			"--hidden-layers", "512",
			"--iterations", "10"
		],
		'subtests': [
			{
				'name': "CPU",
				'executable': "cpu/digits_demo.exe"
			}
		]
	}
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
				print(f"    Test failed with return code {result.returncode}")
				print(f"    stderr: {result.stderr}")
			else:
				print(f"    Test completed successfully in {end_time - start_time:.2f} seconds")
				print(f"    stdout: {result.stdout}")
			time.sleep(1)

