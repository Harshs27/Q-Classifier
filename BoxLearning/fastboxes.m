function [ourtrainingTP,ourtrainingFP,ourtestingTP,ourtestingFP,lowerideal,upperideal]=fastboxes(Apositivetraining,Anegativetraining,Apositivetesting,Anegativetesting,csize,idealk,beta)
%this algorithm take in the expansion parameter beta, 
%the number of cluster, idealk
%Positive training data, Apositivetraining
%Negative training data, Anegativetraining
%Positive test data, Apositivetesting
%Negative test dtata, Anegativetesting (each row is a data point)

%csize, the size that of weight for negative data to try.
%it will enumerate weight from 1/csize, 2/csize, etc... up to 1.

%the algorithms return the true positive and false positive and the
%boundary

%first, make the entries to be less than 1

[mpositive,~]=size(Apositivetraining);
[mnegative,~]=size(Anegativetraining);

[mpositivetesting,~]=size(Apositivetesting);
[mnegativetesting,~]=size(Anegativetesting);

ratiovector=max([abs(Apositivetraining);abs(Anegativetraining)]);

ratioindex=ratiovector>0;
ratiovector=ratiovector(ratiovector>0);
Apositivetraining(:,ratioindex)=Apositivetraining(:,ratioindex)./repmat(ratiovector,mpositive,1);
Anegativetraining(:,ratioindex)=Anegativetraining(:,ratioindex)./repmat(ratiovector,mnegative,1);
Apositivetesting(:,ratioindex)=Apositivetesting(:,ratioindex)./repmat(ratiovector,mpositivetesting,1);
Anegativetesting(:,ratioindex)=Anegativetesting(:,ratioindex)./repmat(ratiovector,mnegativetesting,1);

%normalize all dimension
superupperbound=max(max([Apositivetraining;Anegativetraining;Apositivetesting;Anegativetesting]));
superlowerbound=min(min([Apositivetraining;Anegativetraining;Apositivetesting;Anegativetesting]));

A=[Apositivetraining; Anegativetraining];
overallsize=size(A,1);
overallmean=mean(A);
overallstd=std(A);
tempA=(A-repmat(overallmean,overallsize,1))./repmat(overallstd,overallsize,1);
tempA(:,overallstd==0)=1;

ourtrainingTP=zeros(csize,1);
ourtrainingFP=zeros(csize,1);
ourtestingTP=zeros(csize,1);
ourtestingFP=zeros(csize,1);

%step 2: cluster positive class
[~,n]=size(Apositivetraining);
    
%tempcount=1;
if mpositive==1
    IDX=1; %extreme case of a single sample point
else
    tempD=squareform(pdist(tempA));
    tempstartindex=zeros(1,idealk);
    tempstartindex(1)=unidrnd(mpositive);
    for tempindex=2:idealk
        dummy=setdiff(1:mpositive,tempstartindex(1:tempindex-1));
        [~,dummyparticularindex]=max(sum(tempD(tempstartindex(1:tempindex-1),dummy)));
        tempstartindex(tempindex)=dummy(dummyparticularindex);
    end
    IDX=kmeans(tempA(1:mpositive,:),[],'start',tempA(tempstartindex,:),'emptyaction','drop','onlinephase','off');
end

%just doing some initialization

lowertemppositivesummatrix=zeros(idealk,n);
uppertemppositivesummatrix=zeros(idealk,n);
lowertempnegativesummatrix=zeros(idealk,n);
uppertempnegativesummatrix=zeros(idealk,n);
lowermatrix=zeros(idealk,n);
uppermatrix=zeros(idealk,n);
centermatrix=zeros(idealk,n);
lowerminimalexpansion=zeros(idealk,n);
upperminimalexpansion=zeros(idealk,n);

%next, we do space division

for k=1:idealk  %if you have access to parallel structure, feel free to use parfor, parallel for
    if (isempty(A(IDX==k,:))==0)
    B=A(IDX==k,:);
    temppositiveindex= IDX==k;
    lowermatrix(k,:)=min(B,[],1)-eps;
    uppermatrix(k,:)=max(B,[],1)+eps;
    centermatrix(k,:)=(lowermatrix(k,:)+uppermatrix(k,:))/2;
    lowerindicator=A<repmat(lowermatrix(k,:),overallsize,1); %which points are less than the lower boundary of a box
    for tempj=1:n %feel free to use parfor
        if (sum(lowerindicator(:,tempj))>0)
            lowerminimalexpansion(k,tempj)=0.99*max(A(lowerindicator(:,tempj),tempj))+0.01*lowermatrix(k,tempj); %precompute where is the next negative point
        else
            lowerminimalexpansion(k,tempj)=superlowerbound;  
        end
    end
    centerindicator=A<repmat(centermatrix(k,:),overallsize,1);
    rightinthemiddleindicator=A==repmat(centermatrix(k,:),overallsize,1);
    upperindicator=A>repmat(uppermatrix(k,:),overallsize,1); %which points are higher than the upper boundary of a box
    for tempj=1:n
        if (sum(upperindicator(:,tempj))>0)
            upperminimalexpansion(k,tempj)=0.99*min(A(upperindicator(:,tempj),tempj))+0.01*uppermatrix(k,tempj);
        else
            upperminimalexpansion(k,tempj)=superupperbound;
        end
    end
    
    intheboxindicator=(~lowerindicator) & (~upperindicator);  
    intheboxindicator=all(intheboxindicator,2);
    intheboxindicator=repmat(intheboxindicator,1,n); %which points are inside the box
    lowerflag=(lowerindicator |((centerindicator|rightinthemiddleindicator) & intheboxindicator)); %
    upperflag=(upperindicator |((~centerindicator|rightinthemiddleindicator) & intheboxindicator));
    distancetolowerboundary=A-repmat(lowermatrix(k,:),overallsize,1);
    distancetoupperboundary=A-repmat(uppermatrix(k,:),overallsize,1);
    
    %precompute some statistics
    
    lowertemppositivesummatrix(k,:)=sum((lowerflag(1:mpositive,:).*repmat(temppositiveindex,1,n)).*exp(-distancetolowerboundary(1:mpositive,:)-1));
    weightfactorstorage=(lowerindicator | upperindicator).*exp(-min(abs(distancetolowerboundary),abs(distancetoupperboundary))); %we can precompute this quantity
    weightfactorstorage(weightfactorstorage==0)=1;
    weightfactorstorage=repmat(prod(weightfactorstorage,2),1,n)./weightfactorstorage;
    lowertemppositivesummatrix(lowertemppositivesummatrix>10^16)=10^16;  %numbers that are huge are truncated at 10^16
    lowertemppositivesummatrix(isnan(lowertemppositivesummatrix))=10^16;
   
    expodistancetolowerboundary=exp(distancetolowerboundary+1);
    expodistancetolowerboundary(expodistancetolowerboundary>10^16)=10^16;
    lowertempnegativesummatrix(k,:)=sum(lowerflag.*[repmat(~temppositiveindex,1,n);ones(mnegative,n)].*weightfactorstorage.*expodistancetolowerboundary);
    lowertempnegativesummatrix(isnan(lowertempnegativesummatrix))=10^16;
    
    uppertemppositivesummatrix(k,:)=sum((upperflag(1:mpositive,:).*repmat(temppositiveindex,1,n)).*exp(distancetoupperboundary(1:mpositive,:)-1));
    uppertemppositivesummatrix(uppertemppositivesummatrix>10^16)=10^16;
    uppertemppositivesummatrix(isnan(uppertemppositivesummatrix))=10^16;
    uppertempnegativesummatrix(k,:)=sum(upperflag.*[repmat(~temppositiveindex,1,n);ones(mnegative,n)].*weightfactorstorage.*exp(-distancetoupperboundary+1));
    uppertempnegativesummatrix(uppertempnegativesummatrix>10^16)=10^16;
    uppertempnegativesummatrix(isnan(uppertempnegativesummatrix))=10^16;
    end
end

cvector=(1/csize):(1/csize):1;
cvector=cvector';
lowerideal=zeros(idealk,n);
upperideal=zeros(idealk,n);

for c=1:length(cvector) %feel free to use parfor
    trainingpositiveclassification=zeros(size(Apositivetraining,1),1);
    testingpositiveclassification=zeros(size(Apositivetesting,1),1);
    trainingnegativeclassification=zeros(size(Anegativetraining,1),1);
    testingnegativeclassification=zeros(size(Anegativetesting,1),1);
    for k=1:idealk  %feel free to use parfor
        lowerideal(k,:)=log((-beta+sqrt(beta^2+4*lowertemppositivesummatrix(k,:).*(cvector(c)*lowertempnegativesummatrix(k,:))))./(2*lowertemppositivesummatrix(k,:)));
        templowerideal=lowerideal(k,:);
        templowerideal(:,((-beta+sqrt(beta^2+4*lowertemppositivesummatrix(k,:).*(cvector(c)*lowertempnegativesummatrix(k,:))))./(2*lowertemppositivesummatrix(k,:)))<eps)=log(eps);
        lowerideal(k,:)=templowerideal+lowermatrix(k,:)-1;  %adjustment of boundary
       
        lowerideal(k,(cvector(c)*lowertempnegativesummatrix(k,:))==0)=superlowerbound; %push to far end
        lowerideal(k,:)=min(lowerideal(k,:),lowerminimalexpansion(k,:)); %push towards next negative point
        
        upperideal(k,:)=log((beta+sqrt(beta^2+4*uppertemppositivesummatrix(k,:).*(cvector(c)*uppertempnegativesummatrix(k,:))))./(2*cvector(c)*uppertempnegativesummatrix(k,:)));
        tempupperideal=upperideal(k,:);
        tempupperideal(:,(beta+sqrt(beta^2+4*uppertemppositivesummatrix(k,:).*(cvector(c)*uppertempnegativesummatrix(k,:))))./(2*cvector(c)*uppertempnegativesummatrix(k,:))<eps)=log(eps);
        upperideal(k,:)=tempupperideal+uppermatrix(k,:)+1; %adjustment of boundary
       
        upperideal(k,(cvector(c)*uppertempnegativesummatrix(k,:))==0)=superupperbound; %push to far end
        upperideal(k,:)=max(upperideal(k,:),upperminimalexpansion(k,:)); %push towards next negative point
        
        trainingpositiveclassification=trainingpositiveclassification | all(((Apositivetraining>=repmat(lowerideal(k,:),size(Apositivetraining,1),1)) & (Apositivetraining<=repmat(upperideal(k,:),size(Apositivetraining,1),1))),2);
        testingpositiveclassification=testingpositiveclassification | all(((Apositivetesting>=repmat(lowerideal(k,:),size(Apositivetesting,1),1)) & (Apositivetesting<=repmat(upperideal(k,:),size(Apositivetesting,1),1))),2);
        trainingnegativeclassification=trainingnegativeclassification | all(((Anegativetraining>=repmat(lowerideal(k,:),size(Anegativetraining,1),1)) & (Anegativetraining<=repmat(upperideal(k,:),size(Anegativetraining,1),1))),2);
        testingnegativeclassification=testingnegativeclassification | all(((Anegativetesting>=repmat(lowerideal(k,:),size(Anegativetesting,1),1)) & (Anegativetesting<=repmat(upperideal(k,:),size(Anegativetesting,1),1))),2);
    end

    ourtrainingTP(c)=sum(trainingpositiveclassification);
    ourtrainingFP(c)=sum(trainingnegativeclassification);
    ourtestingTP(c)=sum(testingpositiveclassification);
    ourtestingFP(c)=sum(testingnegativeclassification);
end

%rescale back
lowerideal=lowerideal.*repmat(ratiovector,idealk,1);
upperideal=upperideal.*repmat(ratiovector,idealk,1);

end