%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%               This fast, simple and accurate free space Poisson solver 
%               is based on the article "Fast convolution with free-space
%               Green's function" by F. Vico, L. Greengard, and M.
%               Ferrando, Journal of Computational Physics 323 (2016)
%               191-203
%
%               % Copyright (C) 2018-2021: Junyi Zou and Antoine Cerfon
%               Contact: cerfon@cims.nyu.edu
% 
%               This program is free software; you can redistribute it and/or modify 
%               it under the terms of the GNU General Public License as published by 
%               the Free Software Foundation; either version 2 of the License, or 
%               (at your option) any later version.  This program is distributed in 
%               the hope that it will be useful, but WITHOUT ANY WARRANTY; without 
%               even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
%               PARTICULAR PURPOSE.  See the GNU General Public License for more 
%               details. You should have received a copy of the GNU General Public 
%               License along with this program; if not, see <http://www.gnu.org/licenses/>.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all;

%Initialization
N=60;%number of points in x-domain - must be an even number
L=1.8; %Size of truncation window
h=1/N; %x_step
[x,y,z]=ndgrid(-1/2:h:1/2-h,-1/2:h:1/2-h,-1/2:h:1/2-h); % % Computational grid of N regularly spaced points in each dimension

% Define source for Poisson's problem Delta f = -rho
sigma=0.05;
r2=x.^2+y.^2+z.^2; r=sqrt(r2); %square of radius
rho=1/((2*pi)^(3/2)*sigma^3)*exp(-r2/(2*sigma^2)); % A 3d gaussian distribution

% Exact solutions
% Potential
exact_pot=1./(4*pi*r).*erf(r/sqrt(2)/sigma); %Exact solution to Poisson problem
exact_pot(N/2+1,N/2+1,N/2+1)=sqrt(2)/4/sigma/(pi^(3/2));
% Electric field
exact_Ex = x./(4*pi*r.^2).*(1./r.*erf(r/sqrt(2)/sigma)-sqrt(2/pi)*1/sigma*exp(-r.^2/(2*sigma^2)));
exact_Ex(N/2+1,N/2+1,N/2+1)=0;
exact_Ey = y./(4*pi*r.^2).*(1./r.*erf(r/sqrt(2)/sigma)-sqrt(2/pi)*1/sigma*exp(-r.^2/(2*sigma^2)));
exact_Ex(N/2+1,N/2+1,N/2+1)=0;
exact_Ez = z./(4*pi*r.^2).*(1./r.*erf(r/sqrt(2)/sigma)-sqrt(2/pi)*1/sigma*exp(-r.^2/(2*sigma^2)));
exact_Ex(N/2+1,N/2+1,N/2+1)=0;

% Construct corresponding Fourier domain - with higher resolution to
% account for the need for padding and for the oscillatory nature of the
% kernel, as discussed in the article by Vico et al.
hs=pi/2; %s_step
[wm1,wm2,wm3]=ndgrid(-N*pi:hs:N*pi-hs,-N*pi:hs:N*pi-hs,-N*pi:hs:N*pi-hs); %s-domain

% Compute electric field
sf = pi*[0:N-1, 0, -N+1:-1];% Construct doubly refined Fourier domain for differentiation to get electric field
[sfx,sfy,sfz]=ndgrid(sf,sf,sf);%Wavevectors in two dimensions
%sfx = sfx;
%sfy = sfy;
%sfz = sfz;

% Define high order mollified Green's function
s=sqrt(wm1.^2+wm2.^2+wm3.^2); %radius in s-domain
green=2*(sin(L*s/2)./s).^2;%modified green's function
green(2*N+1,2*N+1,2*N+1)=L^2/2;

% Construct extended domain required for padding
[xx,yy,zz]=ndgrid(-2:h:2-h,-2:h:2-h,-2:h:2-h); %extended x-domain 
constant=exp(-1i*xx*N*pi).*exp(-1i*yy*N*pi).*exp(-1i*zz*N*pi);%precomputation for T

%Precomputation
T1=ifftshift(ifftn(green)).*constant; %precomputation for T
T=T1(N+1:3*N,N+1:3*N,N+1:3*N);% We need T to be defined on [-1,1]^3

% Compute free space solution
result=real(ifftn(fftn(T).*fftn(rho,[2*N 2*N 2*N])));%standard aperiod conv with 2N padding
Ex=-real(ifftn(1i*sfx.*fftn(T).*fftn(rho,[2*N 2*N 2*N])));%standard aperiodic convolution with 2N padding 
Ey=-real(ifftn(1i*sfy.*fftn(T).*fftn(rho,[2*N 2*N 2*N])));%standard aperiodic convolution with 2N padding 
Ez=-real(ifftn(1i*sfz.*fftn(T).*fftn(rho,[2*N 2*N 2*N])));%standard aperiodic convolution with 2N padding 

pot=result(N+1:2*N,N+1:2*N,N+1:2*N); %Potential
Ex=Ex(N+1:2*N,N+1:2*N,N+1:2*N);% x-component of electric field
Ey=Ey(N+1:2*N,N+1:2*N,N+1:2*N);% y-component of electric field
Ez=Ez(N+1:2*N,N+1:2*N,N+1:2*N);% z-component of electric field

% Compute error
relative_errorpot=abs((pot-exact_pot)./exact_pot);
relative_errorEx=abs((Ex-exact_Ex));
relative_errorEy=abs((Ey-exact_Ey));
relative_errorEz=abs((Ez-exact_Ez));


% plot results
[xplot,yplot]=ndgrid(-1/2:h:1/2-h,-1/2:h:1/2-h);


figure(1)
subplot(1,3,1)
plot3(xplot,yplot,pot(:,:,N/2));
title('Potential Numerical');
subplot(1,3,2)
plot3(xplot,yplot,exact_pot(:,:,N/2));
title('Potential Exact');
subplot(1,3,3)
plot3(xplot,yplot,relative_errorpot(:,:,N/2));
title('Potential Relative Error');

figure(2)
subplot(3,3,1)
plot3(x(:,:,N/2),y(:,:,N/2),Ex(:,:,N/2));
title('E_x Numerical');
subplot(3,3,2)
plot3(x(:,:,N/2),y(:,:,N/2),exact_Ex(:,:,N/2));
title('E_x Exact');
subplot(3,3,3)
plot3(x(:,:,N/2),y(:,:,N/2),relative_errorEx(:,:,N/2));
title('E_x Absolute Error');
subplot(3,3,4)
plot3(x(:,:,N/2),y(:,:,N/2),Ey(:,:,N/2));
title('E_y Numerical');
subplot(3,3,5)
plot3(x(:,:,N/2),y(:,:,N/2),exact_Ey(:,:,N/2));
title('E_y Exact');
subplot(3,3,6)
plot3(x(:,:,N/2),y(:,:,N/2),relative_errorEy(:,:,N/2));
title('E_y Absolute Error');
subplot(3,3,7)
plot3(x(:,:,N/2),y(:,:,N/2),Ez(:,:,N/2));
title('E_z Numerical');
subplot(3,3,8)
plot3(x(:,:,N/2),y(:,:,N/2),exact_Ez(:,:,N/2));
title('E_z Exact');
subplot(3,3,9)
plot3(x(:,:,N/2),y(:,:,N/2),relative_errorEz(:,:,N/2));
title('E_z Absolute Error');
