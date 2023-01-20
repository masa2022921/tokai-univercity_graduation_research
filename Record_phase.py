import socket
import pickle
from tkinter import N

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


def letsfft(chnum):
	global volt
	global kazu
	volt2=np.array(volt[1:]).T.tolist()
	global fftkekka
	sampling=128
	ch = volt2[chnum]
	
	chf_kari=[0.0] *sampling
	j=0
	while j<len(ch)/sampling:
		for i in range (sampling):
			k=j*sampling+i
			chf_kari[i]=ch[k]
		chf_kari_np=np.array(chf_kari)
		#chf= DFT(chf_kari_np,0,sampling)
		chf= np.fft.fft(chf_kari_np)
		#print(chf)
		SpectrumAmplitude = [0.0] *sampling
		for i in range(sampling):
			k=j*sampling+i
			SpectrumAmplitude[i] = np.sqrt(
			chf[i].real * chf[i].real + chf[i].imag * chf[i].imag
			)

			fftkekka[k][0] = (i * value_freq) / sampling #Freqency

			fftkekka[k][chnum+1] = SpectrumAmplitude[i]
			

		j=j+1


def DFT(x, nStart, nSample):
	# 信号xをデータnSampleで表される周期関数をみなして計算を行う
	F = [0.0] * nSample
	
	# 各周波数ごとに計算する
	for k in range(0, nSample):	# kを0からnSample-1まで値を1つずつ増やして計算する
		# 数式で表されているシグマの中の計算
		for n in range(0, nSample):
			real = x[n+nStart] * np.cos(-2.0*np.pi*k*n / nSample)
			imag = x[n+nStart] * np.sin(-2.0*np.pi*k*n / nSample)
			F[k] = F[k] + complex(real, imag)
	
	# 計算結果をこのモジュールの戻り値とする
	return F

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
				plt.axis("off")
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
				plt.axis("off")
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
				plt.axis("off")
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
	y = np.floor(np.sin(2*np.pi*herutsu*x)*2048)+2048
	for cnt in range(kazu):
		for it in range(8):
			iv = int(y[cnt])
			iv = format(iv, ' >4')
			byvolt[cnt][it]=iv
			#print(iv)
	byvolt = pickle.dumps(byvolt)
	time.sleep(1.28)
	
	return byvolt


keisokucnt=0


def keisoku():
	global volt
	global keisokucnt
	time.sleep(0.3)
	for i in range(6):
		byvolt = ms.myreceive() #ラズパイとコネクトする
		#byvolt=connectdarui() #コネクト必要ない場合
		print(len(byvolt))
		iv = pickle.loads(byvolt)
		ivint = [[0 for f in range(8)]for i in range(kazu)]
		for cnt in range(kazu):
			for it in range(8):
				ivint[cnt][it]=3.3 *int(iv[cnt][it])/ 4096
		keisokucnt=keisokucnt+1
		volt.extend(ivint)

totalfftkekka0=[[0.0 for f in range(10)]for i in range(1)]
totalfftkekka1=[[0.0 for f in range(10)]for i in range(1)]
totalfftkekka2=[[0.0 for f in range(10)]for i in range(1)]
totalfftkekka3=[[0.0 for f in range(10)]for i in range(1)]


#接続したいときにコメント解除
zyusyo=5655
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect(('192.168.11.20', zyusyo))
ms = MySocket(s)

if __name__ == '__main__':
	try: 
		while True:

			if gazocnt1>maxcnt and gazocnt2>maxcnt and gazocnt3>maxcnt or gazocnt0>maxcnt:
				break



			volt = [[0 for f in range(8)]for i in range(1)]
			fftkekka = [[0.0 for f in range(10)]for i in range(kazu*6)]
			

			fftkiroku()
			if gazocnt1>maxcnt or gazocnt2>maxcnt or gazocnt3>maxcnt or gazocnt0>maxcnt:
				continue
			
			
			letsfft(0)
			letsfft(1)
			letsfft(2)
			letsfft(3)
			letsfft(4)
			letsfft(5)
			letsfft(6)
			letsfft(7)
			if gazoflg==0: #安静時
				print("aa\n")
				totalfftkekka0.extend(fftkekka)

			if gazoflg==1:
				print("a\n")
				totalfftkekka1.extend(fftkekka)

			elif gazoflg==2:
				print("b\n")
				totalfftkekka2.extend(fftkekka)

			elif gazoflg==3:
				print("c\n")
				totalfftkekka3.extend(fftkekka)
	except IndexError:
		fftkekka2=np.array(totalfftkekka0[1:]).T.tolist()
		
		np.savetxt('gazo0_1.csv', totalfftkekka0[1:], delimiter=',')
		np.savetxt('gazo1_1.csv', totalfftkekka1[1:], delimiter=',')
		np.savetxt('gazo2_1.csv', totalfftkekka2[1:], delimiter=',')
		np.savetxt('gazo3_1.csv', totalfftkekka3[1:], delimiter=',')
		#print(len(fftkekka2[0]))
		print(gazocnt0,gazocnt1,gazocnt2,gazocnt3)
	finally:
		np.savetxt('gazo0.csv', totalfftkekka0[1:], delimiter=',')
		np.savetxt('gazo1.csv', totalfftkekka1[1:], delimiter=',')
		np.savetxt('gazo2.csv', totalfftkekka2[1:], delimiter=',')
		np.savetxt('gazo3.csv', totalfftkekka3[1:], delimiter=',')
		print(gazocnt0,gazocnt1,gazocnt2,gazocnt3)
		fftkekka2=np.array(totalfftkekka0[1:]).T.tolist()
		print(len(fftkekka2[0]))



