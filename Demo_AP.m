%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% By Liheng Bian, Oct. 25th, 2016. Contact me: lihengbian@gmail.com.
% This demo does the simulation of Fourier ptychology, and use AP to reconstruct the HR plural image.
% Ref: Liheng Bian et al., "Fourier ptychographic reconstruction using
% Poisson maximum likelihood and truncated Wirtinger gradient".
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all; clc;
addpath(genpath(pwd));

%% FPM simulation parameters
samplename.amplitude = 'Lena_512.png';
samplename.phase = 'Map_512.tiff';

noise.type = 0;
noise.variance = 0;

n = 512; % pixels of high resolution image
M_factor=8; % magnification factor
fprob_flag = 0; % if use aberrant pupil function to generate LR images

newfolder = ['data_reconstruction/Noise_' num2str(noise.type) '_' num2str(noise.variance) '_Probe_' num2str(fprob_flag) '_Amp_' samplename.amplitude '_Phase_' num2str(samplename.phase)];
mkdir(newfolder);

%% generate captured images
[sample, f_sample , im_capture, fprob_real, fprob_save, dkx, dky, kx, ky, Masks] = fun_FPM_Capture(samplename, noise, n, M_factor, fprob_flag);

figure;
subplot(1,2,1),imshow(abs(sample),[]); title('groundtruth amplitude');
subplot(1,2,2),imshow(angle(sample),[]); title('groundtruth phase'); colorbar;

save([newfolder '/sample.mat'],'sample');
save([newfolder '/im_capture.mat'],'im_capture');
save([newfolder '/fprob_real.mat'],'fprob_real');
save([newfolder '/fprob_save.mat'],'fprob_save');
save([newfolder '/dkx.mat'],'dkx');
save([newfolder '/dky.mat'],'dky');
save([newfolder '/kx.mat'],'kx');
save([newfolder '/ky.mat'],'ky');

imwrite(uint8(255*(abs(sample)/max(max(abs(sample))))),[newfolder '/sample_amp.jpg'],'jpg');
im_r_ang = angle(sample)/pi;
im_r_ang = im_r_ang - min(min(im_r_ang));
imwrite(uint8(255*(abs(im_r_ang)/max(max(abs(im_r_ang))))),[newfolder '/sample_phase.jpg'],'jpg');

hwidth = n/M_factor/2;

if fprob_flag == 1
    temp = fprob_real(n/2-hwidth+1:n/2+hwidth,n/2-hwidth+1:n/2+hwidth);
    imwrite(uint8(255*(abs(temp)/max(max(abs(temp))))),[newfolder '/probe_real_amp.jpg'],'jpg');
    im_r_ang = angle(temp)/pi;
    im_r_ang = im_r_ang - min(min(im_r_ang));
    imwrite(uint8(255*(abs(im_r_ang)/max(max(abs(im_r_ang))))),[newfolder '/probe_real_phase.jpg'],'jpg');
end

%% AP reconstruction

newfolder_AP = [newfolder '/AP'];
mkdir(newfolder_AP);
    
alpha = 1;
loopnum1 = 100;
save_sep_AP = 5;

fprob_rec = fprob_save(n/2-hwidth+1:n/2+hwidth,n/2-hwidth+1:n/2+hwidth);

% initializaiton
[n1, n2] = size(sample);
temp = sum(sum(im_capture,1),2);
lr_loc = temp == max(temp);
f_reconst = fftshift(fft2(imresize(sqrt(im_capture(:,:,lr_loc)),[n1, n2]))); % low frequency in center

% calculate the reconstruction error
Relerrs_AP = zeros([1+loopnum1,1]);
Relerrs_AP(1) = norm(f_sample - exp(-1i*angle(trace(f_sample'*f_reconst))) * f_reconst, 'fro')/norm(f_sample,'fro'); % Initial rel. error

% begin iteration
gcf_AP = figure; title('AP'); hold on;
for j = 1:loopnum1
    for i = 1:size(kx,2)
        
        kx1 = n/2+round(kx(1,i)/dkx)-hwidth+1;
        kx2 = kx1+hwidth*2-1;
        ky1 = n/2+round(ky(1,i)/dky)-hwidth+1;
        ky2 = ky1+hwidth*2-1;
        
        phi_j = f_reconst(kx1:kx2,ky1:ky2).*fprob_rec;
        f_phi_j = ifft2(ifftshift(phi_j));
        Phi_j = sqrt(im_capture(:,:,i)).*exp(1j*angle(f_phi_j));
        phi_j_p = fftshift(fft2(Phi_j))*M_factor^2;
        temp = conj(fprob_rec)./(max(max((abs(fprob_rec)).^2))).*(phi_j_p-phi_j);
        f_reconst(kx1:kx2,ky1:ky2) = f_reconst(kx1:kx2,ky1:ky2)+alpha*temp;
    end
    im_reconst = ifft2(ifftshift(f_reconst));
    
    Relerrs_AP(1+j) = norm(f_sample - exp(-1i*angle(trace(f_sample'*f_reconst))) * f_reconst, 'fro')/norm(f_sample,'fro');
    plot(Relerrs_AP(1:1+j));
    pause(0.01);

    if mod(j,save_sep_AP) == 0
        imwrite(uint8(255*(abs(im_reconst))),[newfolder_AP '/recover_AP_amp_' num2str(j) '.jpg'],'jpg');
        imwrite(uint8(255*(abs(im_reconst)/max(max(abs(im_reconst))))),[newfolder_AP '/recover_AP_amp_norm_' num2str(j) '.jpg'],'jpg');
        im_r_ang = angle(im_reconst)/pi;
        im_r_ang = im_r_ang - min(min(im_r_ang));
        imwrite(uint8(255*(abs(im_r_ang)/max(max(abs(im_r_ang))))),[newfolder_AP '/recover_AP_phase_' num2str(j) '.jpg'],'jpg');
        
        fprintf(['AP ' num2str(j) '\n']);
    end

end

save([newfolder_AP '/Relerrs_AP.mat'],'Relerrs_AP');
saveas(gcf_AP, [newfolder_AP '/Relerrs_AP.fig']);

figure;
subplot(1,2,1),imshow(abs(im_reconst),[]); title('AP amplitude');
subplot(1,2,2),imshow(angle(im_reconst),[]); title('AP phase'); colorbar;