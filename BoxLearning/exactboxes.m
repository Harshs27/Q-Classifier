function [MIPtrainingTP, MIPtrainingFP,MIPtestingTP, MIPtestingFP] = exactboxes( Apositivetraining, Anegativetraining, Apositivetesting, Anegativetesting, csize, maxk,cexpand,timeperproblem)
%this is the program to implement exact boxes 
%algorithm.
%the algorithm take in Apositivetraining, the positive training data,
%Anegativetraining, the negative training data
%Apositivetesting, the positive test data
%Anegativetesting, the negative test data
%each row is a data point.

%csize is the number of different weight for negative data point.
%maxk is the cluster size
%cexpand is the parameter to control the number of boxes.
%timeperproblem is the time that you are willing to invest for a single MIP
%problem

%this code requires Gurobi

%gurobi log file will be produced

%also, the training and test True positive and False positive will be
%reported.

v=0.05; %this is the margin size
M=10000; %this is a big number
epsilon=0.00001;  %this is a small number


MIPtrainingTP=zeros(csize,1);
MIPtrainingFP=zeros(csize,1);
MIPtestingFP=zeros(csize,1);
MIPtestingTP=zeros(csize,1);
clear model

A=[Apositivetraining;Anegativetraining];
[mpositive,~]=size(Apositivetraining);
[mnegative,~]=size(Anegativetraining);
[mnegativetesting,n]=size(Anegativetesting);
m=mpositive+mnegative;
%tempA=ones(m,1);
tempA=[ones(mpositive,1); -ones(mnegative,1)];
for k=2:maxk*n
    tempA=sparse(blkdiag(tempA,[ones(mpositive,1); -ones(mnegative,1)]));
end

lconstraintA=sparse([tempA,spalloc(m*n*maxk,n*maxk,0),M*speye(m*n*maxk),spalloc(m*n*maxk,m*n*maxk+m*maxk+m,0)]);
lconstraintA=[-lconstraintA; lconstraintA];

uconstraintA=sparse([spalloc(m*n*maxk,n*maxk,0), tempA,spalloc(m*n*maxk,m*n*maxk,0),-M*speye(m*n*maxk),spalloc(m*n*maxk,m*maxk+m,0)]);
clear tempA
uconstraintA=[uconstraintA; -uconstraintA];

T=repmat(speye(m),1,n);
T=sparse(T);
D=T;
for k=2:maxk
    D=sparse(blkdiag(D,T));
end
clear T
W1=blkdiag(-speye(mpositive),2*n*speye(mnegative));
for k=2:maxk
    W1=sparse(blkdiag(W1,-speye(mpositive),2*n*speye(mnegative)));
end
W2=blkdiag(2*n*speye(mpositive),-speye(mnegative));
for k=2:maxk
    W2=sparse(blkdiag(W2,2*n*speye(mpositive),-speye(mnegative)));
end

yconstraintA=sparse([zeros(2*m*maxk,2*n*maxk),[D,D; -D, -D],[W1;W2],zeros(2*m*maxk,m)]);

clear W1
clear W2
clear D

positivezconstraintA=sparse([spalloc(2*mpositive,(2+2*m)*n*maxk,0),[repmat([eye(mpositive), spalloc(mpositive,mnegative,0)],1,maxk); -repmat([eye(mpositive), spalloc(mpositive,mnegative,0)],1,maxk)],[-maxk*speye(mpositive); speye(mpositive)], spalloc(2*mpositive,mnegative,0)]); 
negativezconstraintA=sparse([spalloc(2*mnegative,(2+2*m)*n*maxk,0),[repmat([zeros(mnegative,mpositive),speye(mnegative)],1,maxk); repmat([spalloc(mnegative,mpositive,0),-speye(mnegative)],1,maxk)],spalloc(2*mnegative,mpositive,0),[maxk*speye(mnegative);-speye(mnegative)]]); 

    model.A=sparse([lconstraintA; uconstraintA; yconstraintA; positivezconstraintA; negativezconstraintA; [speye(n*maxk), -speye(n*maxk), spalloc(n*maxk,2*m*n*maxk+m*maxk+m,0)]]);
    clear lconstraintA
    clear uconstraintA
    clear yconstraintA
    clear positiveconstraintA
    clear negativeconstrraintA
    B=[Apositivetraining; -Anegativetraining];
    B=B(:);
    model.rhs=[repmat(-B+v,maxk,1);repmat(B+(-v+M-epsilon),maxk,1);repmat(B+v,maxk,1);repmat(-B+(-v+M-epsilon),maxk,1);repmat([(2*n-1)*ones(mpositive,1);2*n*ones(mnegative,1)],maxk,1);repmat([zeros(mpositive,1);-ones(mnegative,1)],maxk,1); zeros(2*mpositive,1);maxk*ones(mnegative,1);-ones(mnegative,1); zeros(n*maxk,1)];
    model.sense='<';
    model.vtype=[repmat('C',2*n*maxk,1);repmat('B',2*m*n*maxk+m*maxk+m,1)];
    model.modelsense='max';
    model.lb=[(min(min(A))-1)*ones(2*n*maxk,1);zeros(2*m*n*maxk+m*maxk+m,1)];
    model.ub=[(max(max(A))+1)*ones(2*n*maxk,1);ones(2*m*n*maxk+m*maxk+m,1)];
    tempcount=1;

    for imbalancedc=(1/csize):(1/csize):1
        model.obj=[-cexpand*ones(1,n*maxk),cexpand*ones(1,n*maxk),zeros(1,2*m*n*maxk+m*maxk), ones(1,mpositive), imbalancedc*ones(1,mnegative)];
        clear params;
        params.outputflag=0;
        params.timelimit=timeperproblem;  %this is the time limit, after which, we terminate the MIP solving process and move on to the next problem, the currently found solution will be reported.
	params.outputflag=1;
    params.LogToConsole=0;
	params.logfile=['gurobilog', num2str(imbalancedc),'.txt'];
        result=gurobi(model,params);
        lowerboundary=result.x(1:n*maxk);
        upperboundary=result.x(n*maxk+1:2*n*maxk);
        lowerboundary=reshape(lowerboundary,n,maxk);
        upperboundary=reshape(upperboundary,n,maxk);
        
    lowerideal=lowerboundary';
    upperideal=upperboundary';
    trainingpositiveclassification=zeros(size(Apositivetraining,1),1);
    testingpositiveclassification=zeros(size(Apositivetesting,1),1);
    trainingnegativeclassification=zeros(size(Anegativetraining,1),1);
    testingnegativeclassification=zeros(size(Anegativetesting,1),1);
    
    for k=1:maxk
        trainingpositiveclassification=trainingpositiveclassification | all(((Apositivetraining>=repmat(lowerideal(k,:),size(Apositivetraining,1),1)) & (Apositivetraining<=repmat(upperideal(k,:),size(Apositivetraining,1),1))),2);
        testingpositiveclassification=testingpositiveclassification | all(((Apositivetesting>=repmat(lowerideal(k,:),size(Apositivetesting,1),1)) & (Apositivetesting<=repmat(upperideal(k,:),size(Apositivetesting,1),1))),2);
        trainingnegativeclassification=trainingnegativeclassification | all(((Anegativetraining>=repmat(lowerideal(k,:),size(Anegativetraining,1),1)) & (Anegativetraining<=repmat(upperideal(k,:),size(Anegativetraining,1),1))),2);
        testingnegativeclassification=testingnegativeclassification | all(((Anegativetesting>=repmat(lowerideal(k,:),size(Anegativetesting,1),1)) & (Anegativetesting<=repmat(upperideal(k,:),size(Anegativetesting,1),1))),2);
    end


    TP=sum(testingpositiveclassification);
    FP=sum(testingnegativeclassification);
    TN=mnegativetesting-FP;
    MIPtestingTP(tempcount)=TP;
    MIPtestingFP(tempcount)=mnegativetesting-TN;

    TP=sum(trainingpositiveclassification);
    FP=sum(trainingnegativeclassification);
    TN=mnegative-FP;
    MIPtrainingTP(tempcount)=TP;
    MIPtrainingFP(tempcount)=mnegative-TN;
        
    tempcount=tempcount+1;
    end

end