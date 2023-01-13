#! /usr/bin/python3
#  -*- coding: utf-8 -*-
#
#   mcp3208_read.py
#
#                   Jan/26/2018
#
# --------------------------------------------------------------------
import sys
from gpiozero import MCP3208
import time
import signal
import socket
import pickle
import threading

# --------------------------------------------------------------------
print(sys.getrecursionlimit())

sys.setrecursionlimit(67108864) #64MB
threading.stack_size(1024*1024)  #2の20乗のstackを確保=メモリの確保

#変更後の再帰関数の実行回数の上限を表示
print(sys.getrecursionlimit())


sys.stderr.write("*** 開始 ***\n")
ports = []
kazu = 64


cnt=-1
cnetflg=0
sendcnt=0

zyusyo=5655


for it in range(8):
	ports.append(MCP3208 (channel=it, differential=False))
#
imax = 4096
def yomitori(arg1, args2):
	global cnt
	global volt
	global sendcnt
#	print(cnt)
	for it in range(8):
		iv = int(imax * ports[it].value)
		iv = format(iv, ' >4')
#		print (it,cnt)
		volt[cnt][it]=iv
#		print(volt[cnt][it])
#		print(cnt)
	if cnt >= kazu - 1 :
		byvolt = pickle.dumps(volt)
		c.sendall(byvolt)
#		print(volt)
#		print(len(volt))
		cnt = -1
		sendcnt += 1
	cnt = cnt + 1



s =socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('server start')
s.bind(('',zyusyo))
s.listen(10)
c,addr = s.accept()
cnetflg=cnetflg+1
print('connected')

try:
	if cnetflg  >= 1:
		volt = [[0 for f in range(8)]for i in range(kazu)]
		signal.signal(signal.SIGALRM, yomitori)
		signal.setitimer(signal.ITIMER_REAL, 0.1, 0.02)

		time.sleep(2600)

finally:
#	sys.stderr.write("*** 終了 ***\n")
	s.close()
	print(volt)
	print(cnt)
