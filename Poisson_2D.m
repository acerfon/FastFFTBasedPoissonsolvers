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

computationtype = 'with_pot';%Switch variable to determine if computation of the potential is needed of not. 'with_pot' to compute the electrostatic potential in addition to the components of the electric field; 'no_pot' to only compute the components of the electric field.
sourcetype = 'source2';%Switch variable to choose between two types of sources with an exact solution, for accuracy tests.

%Initialization
% Computational domain

N=100;%number of points in x-domain
L=1.5;%Size of truncation window
h=1/N; %x_step
[x,y]=ndgrid(-1/2:h:1/2-h,-1/2:h:1/2-h); % Computational grid of N regularly spaced points in each dimension

% Define source for Poisson's problem Delta f = -rho
switch sourcetype
    case 'source1'
    sigma=0.05;
    r2=x.^2+y.^2; %square of radius
    rho=1/(2*pi*sigma^2)*exp(-r2/(2*sigma^2)); % A 2d gaussian distribution
    exact_sol = 1/(4*pi)*(ei(-r2/(2*sigma^2))-log(r2));% Exact solution to Poisson problem Delta f = -rho
    exact_sol(N/2+1,N/2+1)=-double(-eulergamma+log(2*sigma^2))/4/pi;
    elec_x_exact = -1/(2*pi)*(x./(x.^2+y.^2)).*(exp(-(x.^2+y.^2)/(2*sigma^2))-1);% Exact x-component of the electric field
    elec_x_exact(N/2+1,N/2+1)=0;
    elec_y_exact = -1/(2*pi)*(y./(x.^2+y.^2)).*(exp(-(x.^2+y.^2)/(2*sigma^2))-1);% Exact y-component of the electic field
    elec_y_exact(N/2+1,N/2+1)=0;
    case 'source2'
    alphax=160;
    alphay=120;
    rho=(2*alphax+2*alphay-4*alphax^2*x.^2-4*alphay^2*y.^2)...
    .*exp(-alphax*x.^2-alphay*y.^2);
    elec_x_exact = 2*alphax*x.*exp(-alphax*x.^2-alphay*y.^2);
    elec_y_exact = 2*alphay*y.*exp(-alphax*x.^2-alphay*y.^2);
    exact_sol = exp(-alphax*x.^2-alphay*y.^2);% Exact solution to Poisson problem Delta f = -rho
end

% Construct corresponding Fourier domain - with higher resolution to
% account for the need for padding and for the oscillatory nature of the
% kernel, as discussed in the article by Vico et al.
hs=pi/2; %s_step for Fourier domain
[wm1,wm2]=ndgrid(-N*pi:hs:N*pi-hs,-N*pi:hs:N*pi-hs);%s-domain is Fourier domain
s=sqrt((wm1).^2+(wm2).^2); %radius in s-domain

% Define high order mollified Green's function
green=(1-besselj(0,L*s))./(s.^2)-(L*log(L)*besselj(1,L*s))./(s);%mollified green's function
green(2*N+1,2*N+1)=L^2/4-L^2*log(L)/2;

% Construct extended domain required for padding
[xx,yy]=ndgrid(-2:h:2-h,-2:h:2-h); %extended x-domain 
constant2=exp(1i*xx*(-N*pi)).*exp(1i*yy*(-N*pi));%precomputation of the convolution operator T

% Precomputation
T1=ifftshift(ifftn(green)).*constant2; %precomputation of the convolution operator T
T=T1(N+1:3*N,N+1:3*N);% We need T to be defined on [-1,1]^2

% Compute free space solution
sf = pi*[0:N-1, 0, -N+1:-1];% Construct doubly refined Fourier domain for differentiation to get electric field
[sfx,sfy]=ndgrid(sf,sf);%Wavevectors in two dimensions

    switch computationtype
        case 'with_pot' %Compute the potential along with the electric field
pot=real(ifftn(fftn(T).*fftn(rho,[2*N 2*N])));%standard aperiodic convolution with 2N padding
pot=pot(N+1:2*N,N+1:2*N);
figure(1)
subplot(3,1,1)
h=pcolor(x,y,pot);
set(h, 'EdgeColor', 'none')
colormap hot
colorbar
title('Potential, numerical')
subplot(3,1,2)
h=pcolor(x,y,exact_sol);
set(h, 'EdgeColor', 'none')
colormap hot
colorbar
title('Potential, exact')
subplot(3,1,3)
h=pcolor(x,y,pot-exact_sol);
set(h, 'EdgeColor', 'none')
colormap hot
colorbar
title('Potential, difference')
        case 'no_pot' %Compute the electric field only
    end
Ex=-real(ifftn(1i*fftn(T).*sfx.*fftn(rho,[2*N 2*N])));%standard aperiodic convolution with 2N padding 
Ey=-real(ifftn(1i*sfy.*fftn(T).*fftn(rho,[2*N 2*N])));%standard aperiodic convolution with 2N padding 
Ex=Ex(N+1:2*N,N+1:2*N);
Ey=Ey(N+1:2*N,N+1:2*N);

figure(2)
subplot(2,3,1)
h=pcolor(x,y,Ex);
set(h, 'EdgeColor', 'none')
colormap hot
title('Ex, numerical')
subplot(2,3,2)
h=pcolor(x,y,elec_x_exact);
set(h, 'EdgeColor', 'none')
colormap hot
title('Ex, exact')
subplot(2,3,3)
h=pcolor(x,y,elec_x_exact-Ex);
set(h, 'EdgeColor', 'none')
colormap hot
colorbar
title('Ex, difference')
subplot(2,3,4)
h=pcolor(x,y,Ey);
set(h, 'EdgeColor', 'none')
colormap hot
title('Ey, numerical')
subplot(2,3,5)
h=pcolor(x,y,elec_y_exact);
set(h, 'EdgeColor', 'none')
colormap hot
title('Ey, exact')
subplot(2,3,6)
h=pcolor(x,y,elec_y_exact-Ey);
set(h, 'EdgeColor', 'none')
colormap hot
title('Ey, difference')
colorbar



                                          
                                          
