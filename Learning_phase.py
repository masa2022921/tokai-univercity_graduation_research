import socket
import pickle
from tkinter import N

from scipy.spatial.distance import mahalanobis
import pandas as pd

import time

import random
from PIL import Image


import numpy as np
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor

#from sklearn import svm

kazu=64



value_freq=50
MSGLEN =3856

t1 = 0/value_freq					# 解析対象データの開始時刻
t2 = t1 + kazu/value_freq
datatime = [0 for f in range(kazu)]
class MySocket:
    """demonstration class only
    - coded for clarity, not efficiency
    """
    

    def __init__(self, sock=None):
        
        print(MSGLEN)
        if sock is None:
            self.sock = socket.socket(
                            socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock

    def connect(self, host, port):
        self.sock.connect((host, port))

    def mysend(self, msg):
        totalsent = 0
        while totalsent < MSGLEN:
            sent = self.sock.send(msg[totalsent:])
            if sent == 0:
                raise RuntimeError("socket connection broken")
            totalsent = totalsent + sent

    def myreceive(self):
        chunks = []
        bytes_recd = 0
        while bytes_recd < MSGLEN:
            chunk = self.sock.recv(min(MSGLEN - bytes_recd, 2048))
            if chunk == b'':
                raise RuntimeError("socket connection broken")
            chunks.append(chunk)
            bytes_recd = bytes_recd + len(chunk)
        return b''.join(chunks)
		



kirokucnt=0
global fftkekka

def letsfft(chnum):
	global volt
	global kazu
	volt2=np.array(volt[1:]).T.tolist()
	global fftkekka
	sampling=128
	ch=[0.0 for f in range(kazu*6)]
	ch = volt2[chnum]
	
	chf_kari=[0.0 for f in range(sampling)]
	j=0
	while j<len(ch)/len(chf_kari):
		for i in range (sampling):
			k=j*sampling+i
			chf_kari[i]=ch[k]
		chf= np.fft.fft(chf_kari)
		
		SpectrumAmplitude = [0.0] *sampling
		for i in range(sampling):
			k=j*sampling+i
			SpectrumAmplitude[i] = np.sqrt(
			chf[i].real * chf[i].real + chf[i].imag * chf[i].imag
			)

			fftkekka[k][0] = (i * value_freq) / sampling #Freqency

			fftkekka[k][chnum+1] = SpectrumAmplitude[i]

		j=j+1


gazoflg=0
gazocnt0=0
gazocnt1=0
gazocnt2=0
gazocnt3=0

maxcnt=50


def fftkiroku():
	global gazoflg
	global gazocnt0
	global gazocnt1
	global gazocnt2
	global gazocnt3
	global keisokucnt

	#gazoNO=random.randint(1, 3)#画像表示
	gazoNO=0 #安静時

	print("b")
	if gazoNO == 0:
		if gazocnt0>=maxcnt:
			return
		with ThreadPoolExecutor(max_workers=4) as e:
			im = Image.open("plane.jpg")
			im_list = np.asarray(im)
			start2=time.time()
			plt.imshow(im_list)
			plt.axis("off")
			plt.draw()
			e.submit(keisoku) 
			while keisokucnt < 6: 
				plt.pause(0.5)
			plt.clf()
			gazoflg=gazoNO
			gazocnt0=gazocnt0+1
			for i in range(kazu):
				fftkekka[i][9] = 0
			keisokucnt= 0


	if gazoNO == 1:
		if gazocnt1>=maxcnt:
			return
		with ThreadPoolExecutor(max_workers=4) as e:
			im = Image.open("sikaku.jpg")
			im_list = np.asarray(im)
			plt.imshow(im_list)
			plt.draw()
			print("b")
			
			e.submit(keisoku) 
			while keisokucnt < 6:
				plt.pause(0.5)
			plt.clf()
			gazoflg=gazoNO
			gazocnt1=gazocnt1+1
			for i in range(kazu):
				fftkekka[i][9] = 1
			keisokucnt= 0


	elif gazoNO == 2:
		if gazocnt2>=maxcnt:
			return
		with ThreadPoolExecutor(max_workers=4) as e:
			im = Image.open("sikaku3.jpg")
			im_list = np.asarray(im)
			plt.imshow(im_list)
			plt.draw()
			print("c")
			e.submit(keisoku) 
			while keisokucnt < 6:
				plt.pause(0.5)
			plt.clf()
			gazoflg=gazoNO
			gazocnt2=gazocnt2+1
			for i in range(kazu):
				fftkekka[i][9]  = 2
			keisokucnt= 0




	elif gazoNO == 3:
		if gazocnt3>=maxcnt:
			return
		with ThreadPoolExecutor(max_workers=4) as e:
			im = Image.open("maru.jpg")
			im_list = np.asarray(im)
			plt.imshow(im_list)
			plt.draw()
			print("d")
			e.submit(keisoku) 
			while keisokucnt < 6:
				plt.pause(0.5)
			plt.clf()
			gazoflg=gazoNO
			gazocnt3=gazocnt3+1
			for i in range(kazu):
				fftkekka[i][9]  = 3
			keisokucnt= 0



def kunren(ch,arg1,arg2):
	with ThreadPoolExecutor(max_workers=4) as e:
		global keisokucnt
		print("OK")
		e.submit(keisoku)
		while keisokucnt < 2:
			time.sleep(0.5)
		global fftkekka
		global hanteikekka
		fftkekkadf=pd.DataFrame(fftkekka,columns=["Freqency","ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8","label"])
		teacher  = pd.read_csv('GAZO_0113_2030_1',header=None,names=["Freqency","ch1","ch2","ch3","ch4","ch5","ch6","ch7","ch8","label"])

		teacher.set_index("Freqency",inplace=True)
		fftkekkadf.set_index("Freqency",inplace=True)

		# 共分散
		cov = np.cov(teacher [ch][arg1], teacher [ch][arg2])

		# 選手AとB
		player_a = [np.mean(fftkekkadf [ch][arg1]), np.mean(fftkekkadf [ch][arg2])]

		# 平均値
		avg = np.array([np.mean(teacher [ch][arg1]), np.mean( teacher [ch][arg2])])

		result1 = 1/(1+mahalanobis(avg, player_a, np.linalg.inv(cov)))
		hantei=[result1,0]
		if result1 <0.2:
		
			im = Image.open("HANTEI_1.PNG")
			im_list = np.asarray(im)
			plt.imshow(im_list)
			plt.axis("off")
			plt.draw()
			while keisokucnt < 4:
				plt.pause(0.5)
			plt.clf()
			hantei[1]=1.0
			print(hantei)
			hanteikekka.append(hantei)

		elif result1 <0.4:
			im = Image.open("HANTEI_2.PNG")
			im_list = np.asarray(im)
			plt.imshow(im_list)
			plt.axis("off")
			plt.draw()

			while keisokucnt < 4:
				plt.pause(0.5)
			plt.clf()
			hantei[1]=2.0
			print(hantei)
			hanteikekka.append(hantei)

		elif result1 <0.6:
				im = Image.open("HANTEI_3.PNG")
				im_list = np.asarray(im)
				plt.imshow(im_list)
				plt.axis("off")
				plt.draw()

				while keisokucnt < 4:
					plt.pause(0.5)
				plt.clf()
				hantei[1]=3.0
				print(hantei)
				hanteikekka.append(hantei)

		elif result1 <0.8:

			im = Image.open("HANTEI_4.PNG")
			im_list = np.asarray(im)
			plt.imshow(im_list)
			plt.axis("off")
			plt.draw()
			while keisokucnt < 4:
				plt.pause(0.5)
			plt.clf()
			hantei[1]=4.0
			print(hantei)
			hanteikekka.append(hantei)
		else:
			im = Image.open("HANTEI_5.PNG")
			im_list = np.asarray(im)
			plt.imshow(im_list)
			plt.axis("off")
			plt.draw()
			while keisokucnt < 4:
				plt.pause(0.5)
			plt.clf()
			hantei[1]=5.0
			print(hantei)
			hanteikekka.append(hantei)
		while keisokucnt < 6:
			time.sleep(0.5)
		volt = [[0 for f in range(8)]for i in range(1)]
		print(volt)
		keisokucnt=0





	
def plotfft(ch,chnum):
	SpectrumAmplitude = SpectrumAmplitude[chnum]
	Freqency = Freqency[chnum]
	global datatime

	for cnt in range(kazu):
		if cnt < kazu-1:
			datatime[cnt+1] = datatime[cnt] + 1/value_freq					# 時刻データ：左端は０
	
	x = ch					# 振幅データ：左から二番目は１

	# 解析対象の関数の波形
	plt.subplot(2,1,1)
	plt.plot(datatime, x, color="b", linewidth=1.0, linestyle="-")
	plt.xlim(datatime[0], datatime[kazu-1])
	plt.ylim(-3.3, 3.3)
	plt.title("signal", fontsize=14, fontname='serif')
	plt.xlabel("Time [s]", fontsize=14, fontname='serif')
	plt.ylabel("Amplitude", fontsize=14, fontname='serif')

	# 振幅スペクトルと位相スペクトルの波形
	plt.subplot(2,1,2)
	plt.plot(Freqency, SpectrumAmplitude, color="b", linewidth=1.0, linestyle="-")
	plt.xlim(0, value_freq/2.0)			
	plt.ylim(0, 600.0)
	#plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])		# 目盛り
	#plt.yticks([0, 100, 200, 300, 400, 500, 600])		# 目盛り
	plt.grid(True) 						# gridを表示する
	plt.title("freqency1 spectrum", fontsize=14, fontname='serif')
	plt.xlabel("freqency1 [Hz]", fontsize=14, fontname='serif')
	plt.ylabel("Amplitude", fontsize=14, fontname='serif')
	
	
	# スペクトル値ピークの周波数を求める：直流成分を除く
	peakFreq = 0.0
	peakValue = 0.0
	for k in range(0, kazu):
		# 0.5Hz成分以上からナイキスト周波数まで
		if Freqency[k] > 0.5 and Freqency[k] < value_freq/2.0:
			# 値がこれまでの最大値より大きければ最大値情報を更新
			if SpectrumAmplitude[k] > peakValue:
				peakFreq = Freqency[k]
				peakValue = SpectrumAmplitude[k]
	
	# 結果を画面に出力する
	print("peak frequency : " + str(peakFreq) + " [Hz]")
	print("heart rate : " + str(60.0*peakFreq) + " [bpm: beat per minute]")

	
	# グラフを画面に表示する
	plt.draw()
	plt.pause(0.0001)
	plt.clf()


def connectdarui():
	byvolt = [[0 for f in range(8)]for i in range(kazu)]
	x = np.arange(0, 1.28, 1/50)
	herutsu=random.randint(5, 10)
	herutsu=10
	y = np.floor(np.sin(2*np.pi*herutsu*x)*2048)
	for cnt in range(kazu):
		for it in range(8):
			byvolt[cnt][it]=y[cnt]
	byvolt = pickle.dumps(byvolt)
	time.sleep(1.28)
	return byvolt


keisokucnt=0


def keisoku():
	global volt
	global keisokucnt
	#time.sleep(0.3)
	for i in range(6):
		byvolt = ms.myreceive() #ラズパイとコネクトする
		#byvolt=connectdarui() #コネクト必要ない場合
		print(len(byvolt))
		iv = pickle.loads(byvolt)
		ivint = [[0 for f in range(8)]for i in range(kazu)]
		for cnt in range(kazu):
			for it in range(8):
				ivint[cnt][it]=5.0 *int(iv[cnt][it])/ 4096
		keisokucnt=keisokucnt+1
		volt.extend(ivint)








if __name__ == '__main__':
	
	#接続したいときにコメント解除
	zyusyo=5655
	s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	s.connect(('192.168.11.17', zyusyo))
	ms = MySocket(s)
	hanteikekka=[]
	while True:

		if gazocnt1>=maxcnt and gazocnt2>=maxcnt and gazocnt3>=maxcnt or gazocnt0>=maxcnt:
			print(hanteikekka)
			np.savetxt('hantei.csv', hanteikekka, delimiter=',')
			break


		volt = [[0 for f in range(8)]for i in range(1)]
		fftkekka = [[0.0 for f in range(10)]for i in range(kazu*6)]

		
		#	print(volt[cnt][0])

		fftkiroku()
		letsfft(0)
		letsfft(1)
		letsfft(2)
		letsfft(3)
		letsfft(4)
		letsfft(5)
		letsfft(6)
		letsfft(7)
		kunren("ch1",6.640625,10.15625)




