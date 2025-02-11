clear;clc;
close all

% End-effector trajectory generation between 2 generic points of the space
p_i = [0.5,-0.3,0.9]';
o_i = pi* [0.2,1.3,-0.4]';
x_i = [p_i;o_i];

p_f = [1.8,1.2,0.8]';
o_f = pi* [0.5,0.5,0.4]';
x_f = [p_f;o_f];

%% Position traj
% Unit vector
v = (p_i - p_f)/norm(p_i - p_f);
v_xy = [v(1);v(2);0];

% New end_point 
distance = 0.05;
p_f_new = p_f + distance *v_xy;
p_f_new(3) = p_f(3);

%% Time law
tf = 20;
ti = 0;
time_step = 10^-3;
time = (0 : time_step : tf);

% Choose the polynomial law degree
% "cubic" -- > pi, pf, vi, vf, tf;
% "quintic" -- > pi, pf, vi, vf, ai, af, tf;

% 1° step: Generic position --> 5 cm to goal
type = "quintic"; % cubic
poly_v = [zeros(1,3),norm(p_f_new - p_i),zeros(1,2),tf];
coeff = trajectory(poly_v,type);
s_p = polyval(coeff,time);

traj_p1 = p_i + s_p .* (p_f_new - p_i)/norm(p_f_new - p_i);

% 2° step: 5 cm to goal --> goal
type = "quintic"; % cubic
poly_v = [zeros(1,3),norm(p_f - p_f_new),zeros(1,2),tf];
coeff = trajectory(poly_v,type);
s_p = polyval(coeff,time);

traj_p2 = p_f_new + s_p .* (p_f - p_f_new)/norm(p_f - p_f_new);

%% Orientation traj
Ri = eultoR(o_i);
Rf = eultoR(o_f);
Rfi = Ri'*Rf;

[theta_tot,r] = angle_axis(Rfi);

type = "quintic"; % cubic
poly_v = [zeros(1,3),theta_tot,zeros(1,2),tf];
coeff = trajectory(poly_v,type);
s_o = polyval(coeff,time);

traj_o = cell([length(time),1]);
for i = 1 : length(time)
    traj_o{i} = Ri * aatoR(s_o(i),r);
end


%% Plot
% Time law
figure;
subplot(3,1,1)
plot(time,traj_p1(1,:));grid on
subplot(3,1,2)
plot(time,traj_p1(2,:));grid on
subplot(3,1,3)
plot(time,traj_p1(3,:));grid on

% Path
s = linspace(0,norm(p_i - p_f));
v_r = cell([3,1]); 

for i = 1:length(p_i)
    v_r{i} = v(i)*s + p_f(i);
end
figure
plot3(p_i(1),p_i(2),p_i(3),'rx'); hold on
plot3(p_f(1),p_f(2),p_f(3),'bx'); grid on
plot3(v_r{1},v_r{2},v_r{3},'c','linew',0.5)

% Unit-vector for new end_point 
s = linspace(0,0.05);
v_r = cell([3,1]); 

for i = 1:length(p_i)
    v_r{i} = v_xy(i)*s + p_f(i);
end

plot3(v_r{1},v_r{2},v_r{3},'k','linew',0.5)
plot3(p_f_new(1),p_f_new(2),p_f_new(3),'gx'); grid on

% trajectories plot
plot3(traj_p1(1,:), traj_p1(2,:), traj_p1(3,:),'r.')
plot3(traj_p2(1,:), traj_p2(2,:), traj_p2(3,:),'k.')

function [coeff] = trajectory(v,type)
%     The v-vector should contain:
%     - "cubic": pi, vi, pf, vf, tf;
%     - "quintic": pi, vi, ai, pf, vf, af, tf;

    if type == "cubic"
        t_end = v(end);
        A_b = [zeros(1,3) 1; zeros(1,2) 1 0; t_end.^(3:-1:0); (3:-1:0).*t_end.^(2:-1:-1)];
        b_b = v(1:end - 1);
        coeff = A_b\b_b;
    else
        if type == "quintic"
             t_end = v(end);
             A_b = [zeros(1,5) 1; 
                   zeros(1,4) 1 0;
                   zeros(1,3) 2 0 0;
                   t_end.^(5:-1:0);
                   (5:-1:0).*t_end.^(4:-1:-1);
                   (5:-1:0).*(4:-1:-1).*t_end.^(3:-1:-2);       
                   ];
            b_b = v(1:end - 1);
%             disp(t_end)
            coeff = A_b\b_b';
        end
    end      
end

function[theta_tot,r] = angle_axis(Ri_f)

theta_tot = acos((sum(diag(Ri_f))-1)/2);

% Caso di singolarità di rappresentazione
if theta_tot == 0
    r = [1;0;0];
elseif theta_tot == pi
    % scegliendo la soluzione positiva
    rx = sqrt((Ri_f(1,1)+1)/2); 
    if rx == 0
        disp('theta_tot=0 ed rx=0')
        disp(Ri_f)
        return;
    end
    ry =(Ri_f(2,1)/(2*rx));
    rz = (Ri_f(3,1)/(2*rx));
    r = [rx;ry;rz];
    r = r/norm(r);
else
    % Non singolarità
    r = 1/(2*sin(theta_tot))*[Ri_f(3,2)-Ri_f(2,3);Ri_f(1,3)-Ri_f(3,1);Ri_f(2,1)-Ri_f(1,2)];
    r = r/norm(r);
end
end

function [RPY] = eultoR(v)

%% Costruzione matrici con angoli RPY 
% Trasformazione da gradi in radianti
% fi = fi*pi/180;
% theta = theta*pi/180;
% psi = psi*pi/180;

fi = v(1);
theta = v(2);
psi = v(3);

RPY = [cos(fi)*cos(theta) cos(fi)*sin(theta)*sin(psi)-sin(fi)*cos(psi) cos(fi)*sin(theta)*cos(psi)+sin(fi)*sin(psi); 
      sin(fi)*cos(theta) sin(fi)*sin(theta)*sin(psi)+cos(fi)*cos(psi) sin(fi)*sin(theta)*cos(psi)-cos(fi)*sin(psi); 
      -sin(theta) cos(theta)*sin(psi) cos(theta)*cos(psi)];

%% Verifica ortonormalità
v = zeros(1,3);
v(1) = RPY(:,1)'*RPY(:,2)<10^-3;
v(2) = RPY(:,3)'*RPY(:,2)<10^-3;
v(3) = norm(RPY(:,1))-norm(RPY(:,2))<10^-3;

if v == zeros(1,3)
    pause
end

end

function [RPY] = aatoR(t,r)

%% Costruzione matrici con angoli RPY 
% Trasformazione da gradi in radianti
% fi = fi*pi/180;
% theta = theta*pi/180;
% psi = psi*pi/180;

RPY = [r(1)^2*(1-cos(t)) + cos(t), r(1)*r(2)*(1-cos(t)) - r(3) * sin(t), r(1)*r(3)*(1-cos(t)) + r(2) * sin(t);
       r(1)*r(2)*(1-cos(t)) + r(3) * sin(t), r(2)^2*(1-cos(t)) + cos(t), r(3)*r(2)*(1-cos(t))- r(1) * sin(t);
       r(1)*r(3)*(1-cos(t)) - r(2) * sin(t), r(3)*r(2)*(1-cos(t)) + r(1) * sin(t), r(3)^2*(1-cos(t)) + cos(t)];

%% Verifica ortonormalità
v = zeros(1,3);
v(1) = RPY(:,1)'*RPY(:,2)<10^-3;
v(2) = RPY(:,3)'*RPY(:,2)<10^-3;
v(3) = norm(RPY(:,1))-norm(RPY(:,2))<10^-3;

if v == zeros(1,3)
    pause
end

end


