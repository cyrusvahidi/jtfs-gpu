import numpy as np
from scipy.io import wavfile
import torch
import resampy
import librosa
import soundfile as sf
import IPython.display as ipd
import pickle

from torch.autograd import backward
from kymatio.torch import TimeFrequencyScattering1D,Scattering1D

import time


def colored_noise(signal):
    N = len(signal)
    phasors = np.exp(2j*np.pi*np.random.rand(N//2-1))
    colored_noise_ft = np.abs(np.fft.fft(signal))+0j
    len_ft = len(colored_noise_ft)
    colored_noise_ft[1:(len_ft//2)] = colored_noise_ft[1:(len_ft//2)]*phasors
    colored_noise_ft[-1:(len_ft-len_ft//2):-1] = colored_noise_ft[-1:(len_ft-len_ft//2):-1]*np.conj(phasors)
    colored_noise = np.fft.ifft(colored_noise_ft)
    return colored_noise

def reconstruct_jtfs(x, 
                lr = 100, 
                n_iter = 200, 
                bold_driver_accelerator = 1.1, 
                bold_driver_brake = 0.55):
    start = time.time()

    err_history = []

    torch.manual_seed(0)
    noise = torch.tensor(np.real(colored_noise(x)),requires_grad=True)
    S_noise = jtfs(noise.cuda())
    
    target = torch.from_numpy(x)
    S_target = jtfs(target.cuda())
    
    t_forward = time.time() #forward processing time
    T_elapsed = [t_forward-start]
    t_start = time.time()
    for i in range(n_iter):
        err = torch.norm(S_noise - S_target)/torch.norm(S_target)

        if i % 1 == 0:
            print('Iteration %3d, loss %.2f' % (i, err.cpu().detach().numpy()))

        err_history.append(err)
        err.backward() # backward pass

        # gradient descent
        delta_y = noise.grad
        with torch.no_grad():
            noise_new = noise - lr * delta_y
        noise_new.requires_grad = True       

        if err_history[i] > err_history[i - 1]:
            lr *= bold_driver_brake
        else:
            lr *= bold_driver_accelerator
            noise = noise_new

        S_noise = jtfs(noise) # forward pass
        t_end = time.time()
        T_elapsed.append(t_end-t_start)
        t_start = time.time()

    end = time.time()
    print(end-start)
    return target, noise,err_history,T_elapsed


def reconstruct_timesc(x, 
                lr = 100, 
                n_iter = 200, 
                bold_driver_accelerator = 1.1, 
                bold_driver_brake = 0.55):
    start = time.time()
    err_history = []

    torch.manual_seed(0)
    noise = torch.randn((N,), requires_grad=True)
    S_noise = timesc(noise.cuda())
    
    target = torch.from_numpy(x)
    S_target = timesc(target.cuda())


    for i in range(n_iter):
        err = torch.norm(S_noise - S_target)/torch.norm(S_target)

        if i % 10 == 0:
            print('Iteration %3d, loss %.2f' % (i, err.cpu().detach().numpy()))

        err_history.append(err)
        err.backward() # backward pass

        # gradient descent
        delta_y = noise.grad
        with torch.no_grad():
            noise_new = noise - lr*delta_y
        noise_new.requires_grad = True       

        if err_history[i] > err_history[i - 1]:
            lr *= bold_driver_brake
           
        else:
            lr *= bold_driver_accelerator
            noise = noise_new

        S_noise = timesc(noise) # forward pass

    end = time.time()
    print(end-start)
    return target, noise,err_history

if __name__ == "__main__":
    filename = "../audio/media_accipiter.wav"
    x,sr = sf.read("../audio/"+filename) #librosa.load("../audio/"+filename,sr=22050)
    if len(x.shape)>1:
        x = np.mean(x,axis=1)
    T = [13]#,14,15]
    q = 12
    j = 12

    N = 2**int(np.floor(np.log(len(x))/np.log(2)))
    reconstructed_jtfs = {}
    reconstructed_timesc = {}
    err_jtfs_histories = {}
    err_timesc_histories = {}

    for t in T:
        jtfs = TimeFrequencyScattering1D(
            J = j, #scale, how big the biggest time support (for the lowest freqeuncy), where center frequencies are?
            shape = (N, ), 
            Q = q, #filters per octave, frequency resolution
            T = 2**t, 
            max_pad_factor=1,
            max_pad_factor_fr=1,
            average_fr = False,
        ).cuda()
        _, recon_jtfs,errs_jtfs,t_jtfs = reconstruct_jtfs(x[:N])
        reconstructed_jtfs[(j,q,t)] = recon_jtfs
        err_jtfs_histories[(j,q,t)] = errs_jtfs

        timesc = Scattering1D(
	        J = j, #scale, how big the biggest time support (for the lowest freqeuncy), where center frequencies are?
	        shape = (N, ),
	        Q = q,
	        T = 2**t, 
	        max_order=2,
	        max_pad_factor=1,
	        ).cuda()
        _, recon_timesc,errs_timesc = reconstruct_timesc(x[:N])
        reconstructed_timesc[(j,q,t)] = recon_timesc
        err_timesc_histories[(j,q,t)] = errs_timesc

    for key,val in reconstructed_jtfs.items():
        j,q,t = key
        sf.write("reconstructed_jtfs_gull_j"+str(j)+"_q"+str(q)+"_t"+str(t)+".wav",val.detach().numpy(),sr)
    
    for key,val in reconstructed_timesc.items():
        j,q,t = key
        sf.write("reconstructed_timesc_gull_j"+str(j)+"_q"+str(q)+"_t"+str(t)+".wav",val.detach().numpy(),sr)
    

    #with open('reconstruction_acc_error.pickle', 'wb') as handle:
   # 	pickle.dump(err_jtfs_histories, handle, protocol=pickle.HIGHEST_PROTOCOL)
   # with open("reconstruction_acc_time.pickle","wb") as handle:
    #    pickle.dump(t_jtfs,handle, protocol=pickle.HIGHEST_PROTOCOL)
   # with open('reconstructed_timesc_acc.pickle', 'wb') as handle:
   # 	pickle.dump(err_timesc_histories, handle, protocol=pickle.HIGHEST_PROTOCOL)







