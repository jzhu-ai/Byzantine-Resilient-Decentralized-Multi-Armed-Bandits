err = zeros(k,1);    % reward loss at each time 
n_pull = zeros(n,d); % sample count
bar = zeros(n,d);    % sample mean
idx = zeros(n,d);
at = zeros(n,1); % pulled arm
err_agg = zeros(k,1);
tildef = zeros(n,d,k);
upper = zeros(k,1);

for i = 2:n                        % initialization for normal agent
    for j = 1:d
        bar(i,j) = X(i,j,k-j+1);     
        n_pull(i,j) = 1;
        state(i,j) = bar(i,j);
        idx(i,j) = state(i,j);
    end
end

for t = 1:(k-d)
    for j = 1:d
        n_pull(1,j)=max(n_pull(2:n,j)); % Byzantine updating policy
    end
    for i = f+1:n   % normal agents
        for j = 1:d
            A = [];
            for h = 1:n
                if h ~= i && kappa * n_pull(h,j)>= n_pull(i,j) && MA(i,h) == 1  % consisitency filter
                    A = [A,h];   
                end
            end
            if length(A) <= 2 * f
                state(i,j) = bar(i,j);
                g = 1;
            else
               [order,~]=sort(bar(A,j));
                s1 = sum(order(1:f))+sum(order(length(A)-f+1:length(A)));
                state(i,j) = (sum(bar(A,j))+bar(i,j)-s1)/(length(A)-2*f+1);   % trimmed-mean
                eta = 1/(length(A)-2*Bn+1);
                g = (4*eta^2+kappa*eta+kappa)/4;   
            end
            tildef(i,j,t) = f/(mu(1)-mu(j));
            idx(i,j) = state(i,j)+(2*g*log(t)/n_pull(i,j))^.5;   % confidence variable
        end
        [~, order_arm] = sort(idx(i,:), 'descend');   % choose the arm with the largest confidence
        at(i) = order_arm(1);
        bar(i, at(i)) = (bar(i, at(i)) * n_pull(i,at(i)) + X(i,at(i),t))/(n_pull(i,at(i))+1);  
        n_pull(i,at(i)) = n_pull(i,at(i)) + 1; 
        err(t) = err(t)+abs(mu(at(i))-max(mu)); 
    end
     err(t) = err(t)/(n-Bn);   % network averaged reward loss
end
for t = 1:k-d
    %for i = 2:n
    %err_agg(t,i) = sum(err_in(1:t,i));
    err_agg(t) = sum(err(1:t));   % accumulated network averaged reward loss
    if t == 1
        upper(t) = 0;
    else
    upper(t) = (8*max(sum(sum(tildef(1+Bn:n,2:d,t)))*log(t),upper(t-1)))/(n-Bn);
    end
    %end
end
upper = upper + ones(k-d,1) * (1+pi^2/3)*(d*mu(1)-sum(mu(1:d)));   % upper bound