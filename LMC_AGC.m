function [P, H, S, W,obj] = MCAGC_new(data, alpha, sigma,maxIter, K)
% 原始数据data，维度为1xview_num view_num个视图,是元胞数组
% alpha 为超参数
% sigma 为高斯核
% output 
% P 潜在表示模型，用于重建原始数据 维度为dxview_num
% H 潜在表示，维度为view_numxn
% S 相似度矩阵，由高斯核构造 维度为 nxn
% W 权重矩阵 维度为view_numxn
%%%%%%%%%%%%%%%%%%%%%
% start parpool by codes.
delete(gcp('nocreate'));         % stop the process before start a new run.
numCore = feature('numcores');   % get the maxmium core num of PC.
parpool(numCore-1);              % start parpool.
%%%%%%%%%%%%%%%%%%%%%
view_num = size(data,2);    %视图个数
k = K;                    %H的维度
% for i=1:view_num
%     data{i} = data{i}./repmat(sqrt(sum(data{i}.^2,1)),size(data{i},1),1); %标准化
% end
X=[];
for i=1:view_num
    X = vertcat(X, data{i});            %所有数据d*n
end
[d, n] = size(X);
X=NormalizeFea(X,0);
%% Initialize input parameters

W = ones(view_num, n)./view_num;%% view_num*n
P = zeros(d,k);                 %% d*k
S = zeros(n,n);                 %% n*n
H = zeros(k,n);                 %% k*n
obj = zeros(maxIter, 1);        %% maxIter*1

%% Initialize similarity matrix for each view
options = [];
options.NeighborMode = 'KNN';
options.k =10;                
    
options.WeightMode = 'HeatKernel';
options.t = 1;                  % Kernel sigma

for v=1:view_num
    W = constructW(data{v}',options);
    Affinity = full(W);
    S_d.data{v}=Affinity;
end
clear W;
W = constructW(X',options);
Affinity = full(W);
S=Affinity;
clear W options;
% S_temp = S;
% for v = 1:view_num
%     data_v = data{v};
%     for i = 1:n
%         for j = 1:n
%             if i~=j
%                 S_temp(i,j) = exp(-(sum((data_v(:,i) - data_v(:,j)).^2))/sigma);
%             end
%         end
%         S_temp(i,:) = S_temp(i,:)./sum(S_temp(i,:));
%     end
%     S_d.data{v}=(S_temp+S_temp')./2;
% end
% clear S_temp indx data_v;
% 
% %% Initialize global similarity matrix
% for i = 1:n
%     for j = 1:n
%         if i~=j
%             S(i,j) = exp(-(sum((X(: , i) - X(: , j)).^2))/sigma);
%         end
%     end
%     S(i,:) = S(i,:)./sum(S(i,:));
% end
% S=(S+S')./2;
% W = ones(view_num, n)./view_num;
% S = zeros(n,n); 
iter=1;err=1;
%% 迭代收敛
while (err>0.001&&iter<maxIter)
     %% update W
    disp('update W...');
    parfor i = 1:n
        A_i = zeros(n, view_num);
        for v = 1:view_num
            A_i(:,v) = S(:,i)-S_d.data{v}(:,i);
        end
        part_bi = A_i'*A_i ;
        part_1v = ones(view_num,1);
        temp_inv = part_bi \ part_1v;
        W(:,i) = temp_inv / (part_1v' * temp_inv +1e-15);
    end
    clear A_i part_bi part_1v temp_inv;

      %% update S
%     disp('update S...');
    parfor i = 1:n
        B_i = zeros(n, view_num);
        for v = 1:view_num
            B_i(:,v) = S_d.data{v}(:,i);
        end
%         a_i = zeros(n, 1);
%         for p = 1:n
%             a_i(p) = norm(H(:,i)-H(:,p), 'fro')^2;
%         end
        a_i = sum((H - H(:,i)).^2, 1);
        a_i = a_i';
        part_m = B_i * W(:,i) - 0.25 * alpha * a_i;
        %disp('update psi...');
        psi = zeros(n, 1);
        temp = part_m - ones(n,n) * part_m / n + 1/n - 0.5 * mean(psi);
%         for j = 1:n
%              psi(j) = max(-2*temp(j), 0);
%         end
        psi = max(-2 * temp, 0);
        temp = part_m - ones(n,n) * part_m / n + 1/n - 0.5 * mean(psi);
%         for j = 1:n
%             S(i, j) = max(temp(j), 0);
%         end
         S(i,:) = max(temp', 0);
    end
    clear B_i a_i part_m psi temp temp_psi err2;


     %% update P
%     disp('update P...');
    temp=H*X';
    [U_svd,~,V_svd]=svd(temp,'econ');
    PT = U_svd*V_svd';
    P = PT';
    clear temp temp1 U_svd S_svd V_svd;
    

    %% update H
%     disp('update H...');
    LapMatrix = diag(sum(S, 2)) - S;
    A=2 * (P') * P;
    B=alpha * (LapMatrix + LapMatrix');
    C=2* (P') * X;
    H = sylvester(A,B,C);
    clear temp temp1 A B C;
  
    %% calculate objective function value1
%     disp('calculate obj-value...');
    %亲和矩阵重构损失
    temp_formulation1 = 0;
    parfor i =1:n
        temp_S_i = zeros(n,1);
        for v = 1:view_num
            temp_S_i = temp_S_i + W(v,i)*S_d.data{v}(:,i);
        end
        temp_formulation1 = temp_formulation1 + norm(S(:,i)-temp_S_i,'fro')^2;
    end
    %潜在表示重构损失
    temp_formulation2 = norm(X-P*H,'fro')^2;
    %结构保持损失
    LapMatrix = diag(sum(S, 2)) - S;
    temp_formulation3 = alpha * trace(H * LapMatrix * H');
    obj(iter) = temp_formulation1 + temp_formulation2 + temp_formulation3;
    clear temp_formulation1 temp_S_i temp_formulation2 temp_formulation3;
    if iter>1
        err = abs(obj(iter-1)-obj(iter));
    end
    iter = iter+1;
end
end


