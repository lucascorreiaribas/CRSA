function [features] = CRSA(im,thresholding,Q,W)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Complex Representation for Shape Analysis (CRSA) by Lucas C. Ribas
%
% Input:
%       im: imagem 0=background and >0 contour points
%       thresholding: set of threshold values (0.025:0.005:0.95)
%       Q: set of numbers of hidden neurons ([9,19,59])
%       W: set of window size = [4,7];
%
% Output:
%       features: computed features for shape as Complex Representation
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

feask = []; feasf = [];

[degree,force] = CNDescriptors(im,thresholding);

for r = 1:length(W)
    [Xk,Dk] = windows(degree,degree,W(r));
    [Xf,Df] = windows(force,degree,W(r));
    q = 1;
    for QQ = Q
        M = ELM(Xk,Dk,QQ);
        feask = [feask M];
        
        M = ELM(Xf,Df,QQ);
        feasf = [feasf M];
    end
end

features = [feask feasf];
end


%--------------------------------------------
function [grade,force] = CNDescriptors(img,thresholding)

[coord(:,1) coord(:,2)] = find(img>0);

%network modeling
cn = cnMake(coord);

%CN evoluting with the thresholding selection
[grade,force] = thresholdingCN(cn,thresholding);

%grade normalizadion
grade = grade ./ length(grade);

force = force ./ length(force);

end

%--------------------------------------------
function [CN] = cnMake(coord)

[nx ny] = size(coord);

y = pdist(coord,'euclidean');
CN = squareform(y);

CN = CN ./ max(CN(:));

end

%--------------------------------------------
function [grade,force] = thresholdingCN(cn,thre)

cnU = zeros(size(cn));
cnW = zeros(size(cn));
grade = [];
force = [];
for x=1:length(thre)
    c = cn < thre(x);
    cnU(c) = 1;
    cnW(c) = cn(c);
    grade(x,:) = sum(cnU) - 1;
    force(x,:) = sum(cnW);
end

end

%--------------------------------------------
function [X,D] = windows(mat1,mat2,n)

X = []; D = [];
for i = 1:(size(mat1,1) - n)
    X = [X mat1(i:(i+n-1),:)];
    D = [D mat2((n+i),:)];
end

end

%--------------------------------------------
function [M] = ELM(X,D,Q)

[P N] = size(X); %P is the number of input and N the number of samples

%normalization Z-Score
X = zscore(X');
X = X';
W = LCG(Q,P+1,Q*(P+1)); % matrix of weights
X = [X;-ones(1,N)]; %add bias

Z = g(W*X); %function of activation of hidden layear
Z = [Z -ones(1,N)']'; %add bias hidden output

lambda = 0.001;
M = (D*Z')/(Z*Z'+ lambda * eye(Q+1)); % weights of output layear

end

%--------------------------------------------
%% sigmoide fuction
function y = g(u)
y = 1./(1+exp(-u))';
end

%--------------------------------------------
%% LCG Random
function Mx = LCG(m,n,L)
V(1)=L+1;
a = L+2;
b = L+3;
c = L^2;
for x=1:(m*n)-1
    V(x+1) = mod((a*V(x)+b),c);
end

V = zscore(V);
Mx = reshape([V(:) ; zeros(rem(n - rem(numel(V),n),n),1)],n,[]).';
end


