start /B python Modifytest_fft.py --snapshot .\snapshots\3_100_epoch_400.pth -—nc 100 >> test11.txt
start /B python Modifytest_fft.py --snapshot .\snapshots\3_200_epoch_400.pth -—nc 200 >> test12.txt
start /B python Modifytest_fft.py --snapshot .\snapshots\3_400_epoch_400.pth -—nc 400 >> test13.txt
start /B python Modifytest_fft.py --snapshot .\snapshots\3_800_epoch_400.pth -—nc 800 >> test14.txt
start /B python Modifytest_fft.py --snapshot .\snapshots\3_1600_epoch_400.pth -—nc 1600 >> test15.txt
start /B python Modifytest_fft.py --snapshot .\snapshots\3_3200_epoch_400.pth -—nc 3200 >> test16.txt