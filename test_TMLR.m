clear
clc
d = 4;   %number of arms
mu = [0.5, 0.45, 0.4, 0.3];   %reward mean
%mu = rand(d,1);
%mu = sort(mu, 'descend'); 
n = 10;   %number of agents
k = 10000;   %total time
X = zeros(n,d,k+d); 
run = 50;
x = 1:100:k;
err_l = ones(k,1)*k;
err_r = zeros(k,1);
err_average = zeros(k,1);
f = 1;   % number of Byzantine agents 
for r = 1:run
    for i = 1:n
        for j = 1:d
            for t = 1:k+d
                temp = rand;
                if temp < mu(j)
                   X(i,j,t) = 1;  %reward follows Bernoulli distribution
                else
                   X(i,j,t) = 0;
                end         
            end
        end
     end

state = zeros(n,d);   % reward mean estimate z
for i  = 1 : f
    state(i,:) = [0.4,0.5,0.4,0.3];  % tunable Byzantine policy for updating reward mean estimate
end

MA = ones(d);   % communication matrix, 1 means having an edge, 0 otherwise, the example is for a complete graph

kappa = 1.5;  % tunable parameter kappa
dist_TMLR;
err_l = min(err_l, err_agg);
err_r = max(err_agg, err_r);
err_average = ((r - 1)* err_average + err_agg)/r;

end
plot(x,err_average(1:100:k),':r','LineWidth', 2);
hold on
X=[x,fliplr(x)];  
Y1 = [err_l(1:100:k)', fliplr(err_r(1:100:k)')];
fill(X,Y1,'r','EdgeAlpha', 0,'FaceAlpha',0.1)
legend('resilient UCB','location','best')
xlabel('time')
ylabel('average regret')
grid on

