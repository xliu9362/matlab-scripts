N=129; %number of points
%{
pinwheel location
--------------------------
          y axis
            ^
 --  -   |  |  |   +   --
            |
------------|-------------> x axix
            |
 --  +   |  |  |   -   --   

---------------------------
'-' sign stands for clockwise rotation
'+' sign stands for counter clockwise rotation

%}
%==========================================================================
% STEP 1: Compute the right-top pinwheel matrix
% Size of the fundamental domain 
X_rt=(0:1/(N-1):1);
Y_rt=(0:1/(N-1):1);

% Pinwheel centre
x_pw = 0.5; % mm
y_pw = 0.5; % mm


% Shift coordinates
Xc = X_rt - x_pw;
Yc = Y_rt - y_pw;

%%

% Grid of the domain
[X, Y] = meshgrid(Xc, Yc);

% theta
A        = atan2d(Y, X);

xy_idx = ceil(N/2);
A_rt(xy_idx, xy_idx) = -inf;

figure(1);
imagesc(Xc, Yc, A); hold on;
title('\theta(x, y) = atan2d((y-y_c) / (x-x_c))')
colormap('hsv')
axis equal
xlim([-0.5, 0.5])
ylim([-0.5, 0.5])

xlabel('x-x_c [mm]')
ylabel('y-y_c [mm]')
set(gca, 'YDir', 'normal');
colorbar()
%%

% Shift range to 0 - 180 
% phi = 1/2 * (theta + 180)
A_rt = 0.5*(atan2d(Y, X) + 180);

figure(2);
imagesc(X_rt, Y_rt, A_rt); hold on;
title('\phi(x, y) = 0.5*(\theta + \pi)')
colormap('hsv')
scatter(x_pw, y_pw, 50, [1 1 1], 's', 'filled');
axis equal
xlim([-0.0, 1.0])
ylim([-0.0, 1.0])

xlabel('x [mm]')
ylabel('y [mm]')

% This command basically flips the array upside down
set(gca,'YDir','normal')

colorbar()
%%  Test rotation symmetry
% figure(3);
% subplot(2,2,2)
% h = pcolor(X_rt, Y_rt, A_rt); hold on;
% title('\phi(x,y)')
% set(h,'EdgeColor','none')
% colormap('hsv')
% scatter(X_rt(xy_idx), Y_rt(xy_idx), 50, [1 1 1], 's', 'filled');
% axis equal
% xlim([0, 1])
% ylim([0, 1])
% 
% rot_angle = 1; 
% subplot(2,2,1)
% h1 = pcolor(X_rt, Y_rt, rot90(A_rt,rot_angle)); hold on;
% title('\phi(x,y)')
% 
% set(h1,'EdgeColor','none')
% colormap('hsv')
% scatter(X_rt(xy_idx), Y_rt(xy_idx), 50, [1 1 1], 's', 'filled');
% axis equal
% xlim([0, 1])
% ylim([0, 1])
% ylabel('y [mm]')
% 
% 
% subplot(2,2,3)
% h2 = pcolor(X_rt, Y_rt, rot90(rot90(A_rt,rot_angle),rot_angle)); hold on;
% set(h2,'EdgeColor','none')
% colormap('hsv')
% scatter(X_rt(xy_idx), Y_rt(xy_idx), 50, [1 1 1], 's', 'filled');
% axis equal
% xlim([0, 1])
% ylim([0, 1])
% 
% ylabel('y [mm]')
% xlabel('x [mm]')
% 
% 
% subplot(2,2,4)
% h3 = pcolor(X_rt, Y_rt, rot90(rot90(rot90(A_rt,rot_angle),rot_angle),rot_angle)); hold on;
% set(h3,'EdgeColor','none')
% colormap('hsv')
% scatter(X_rt(xy_idx), Y_rt(xy_idx), 50, [1 1 1], 's', 'filled');
% axis equal
% xlim([0, 1])
% ylim([0, 1])
% 
% xlabel('x [mm]')

%% Test rotation symmetry

% figure(4);
% 
% rot_angle = -1; 
% subplot(2,2,1)
% h1 = pcolor(X_rt, Y_rt, rot90(A_rt,rot_angle)); hold on;
% set(h1,'EdgeColor','none')
% colormap('hsv')
% scatter(X_rt(xy_idx), Y_rt(xy_idx), 50, [1 1 1], 's', 'filled');
% axis equal
% xlim([0, 1])
% ylim([0, 1])
% ylabel('y [mm]')
% 
% 
% subplot(2,2,2)
% h = pcolor(X_rt, Y_rt, A_rt); hold on;
% title('\phi(x,y)')
% set(h,'EdgeColor','none')
% colormap('hsv')
% scatter(X_rt(xy_idx), Y_rt(xy_idx), 50, [1 1 1], 's', 'filled');
% axis equal
% xlim([0, 1])
% ylim([0, 1])
% 
% 
% 
% subplot(2,2,3)
% h2 = pcolor(X_rt, Y_rt, rot90(rot90(A_rt,rot_angle),rot_angle)); hold on;
% set(h2,'EdgeColor','none')
% colormap('hsv')
% scatter(X_rt(xy_idx), Y_rt(xy_idx), 50, [1 1 1], 's', 'filled');
% axis equal
% xlim([0, 1])
% ylim([0, 1])
% 
% ylabel('y [mm]')
% xlabel('x [mm]')
% 
% 
% subplot(2,2,4)
% h3 = pcolor(X_rt, Y_rt, rot90(rot90(rot90(A_rt,rot_angle),rot_angle),rot_angle)); hold on;
% set(h3,'EdgeColor','none')
% colormap('hsv')
% scatter(X_rt(xy_idx), Y_rt(xy_idx), 50, [1 1 1], 's', 'filled');
% axis equal
% xlim([0, 1])
% ylim([0, 1])
% 
% xlabel('x [mm]')
% 
%% Let's flip things around

figure(5);

subplot(2,2,1)
imagesc(X_rt, Y_rt, fliplr(A_rt)); hold on;
title('fliplr (\phi(x,y))')
colormap('hsv')
scatter(x_pw, y_pw, 50, [1 1 1], 's', 'filled');
axis equal
xlim([0, 1])
ylim([0, 1])
ylabel('y [mm]')
set(gca, 'YDir', 'normal')
colorbar()


subplot(2,2,2)
imagesc(X_rt, Y_rt, A_rt); hold on;
title('\phi(x,y)')
colormap('hsv')
scatter(x_pw, y_pw, 50, [1 1 1], 's', 'filled');
axis equal
xlim([0, 1])
ylim([0, 1])
set(gca, 'YDir', 'normal')
colorbar()


subplot(2,2,3)
imagesc(X_rt, Y_rt, rot90(rot90(A_rt))); hold on;
title('rotate 180 (\phi(x,y))')
colormap('hsv')
scatter(x_pw, y_pw, 50, [1 1 1], 's', 'filled');
axis equal
xlim([0, 1])
ylim([0, 1])

ylabel('y [mm]')
xlabel('x [mm]')
set(gca, 'YDir', 'normal')
colorbar()


subplot(2,2,4)
imagesc(X_rt, Y_rt, flipud(A_rt)); hold on;
title('flipud (\phi(x,y))')
colormap('hsv')
scatter(x_pw, y_pw, 50, [1 1 1], 's', 'filled');
axis equal
xlim([0, 1])
ylim([0, 1])
xlabel('x [mm]')
set(gca, 'YDir', 'normal')
colorbar()

%% Make flipped and rotated copies

% To get the unit cell in the right direction we need to flip the
% fundamental domain
%     y|
%      |
%      |
% (0,0) ---->x
A_rt_flipped = flipud(A_rt);

tr = A_rt_flipped;
tl = fliplr(A_rt_flipped);
bl = rot90(rot90(A_rt_flipped));
br = flipud(A_rt_flipped);

%% Concatenate

tr = tr(1:end-1, 2:end);
tl = tl(1:end-1, :); % This one keeps the origin
% bl does not change
br = br(:, 2:end);
PMM = [tl tr; bl br]; 

% Length of primitive cell
X_pc = linspace(-1,1,size(PMM,1)); 
Y_pc = linspace(-1,1,size(PMM,1)); 
[XX, YY] = meshgrid(X_pc, Y_pc);


%%

figure(6)
imagesc(X_pc, Y_pc, PMM); hold on;
colormap('hsv')
axis equal
xlim([-1.0, 1.0])
ylim([-1.0, 1.0])
plot([-1.0, 1.0], [0, 0], 'k')
plot([0.0, 0.0], [-1.0, 1.0], 'k')
plot([-1.0, 1.0], [1.0, -1.0], 'k--')
plot([-1.0, 1.0], [-1.0, 1.0], 'k--')
ylabel('y [mm]')
xlabel('x [mm]')
set(gca, 'YDir', 'normal')
colorbar()

%%

XC = [-0.8, -0.5, -0.5, -0.25, -0.8, -0.70,  -0.3,  -0.2, -0.5,   0];
YC = [-0.5, -0.8, -0.2, -0.3, -0.7,  -0.25,  -0.7, -0.5,     0, -0.7];
TH = [  90,  135,   45,   15,  105,     65,  155,     0,    45, 170];

for kk=1:length(XC)
    % Lines at 90
    [xcoo, ycoo] = oriented_segment(XC(kk), YC(kk), -TH(kk), 0.05);
    line(xcoo, ycoo, 'color', 'k', 'linewidth', 2)
    line(xcoo, ycoo+2*abs(YC(kk)),  'color', 'k', 'linewidth', 2)
    line(xcoo+2*abs(XC(kk)), ycoo,  'color', 'k', 'linewidth', 2)
    line(xcoo+2*abs(XC(kk)), ycoo+2*abs(YC(kk)),  'color', 'k', 'linewidth', 2)
end
