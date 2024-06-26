classdef MT_KRVEA_Pareto < Algorithm
% <Multi-task> <Multi-objective> <None/Constrained>
% Surrogate-assisted RVEA
% alpha ---  2 --- The parameter controlling the rate of change of penalty
% wmax  --- 20 --- Number of generations before updating Kriging models
% mu    ---  5 --- Number of re-evaluated solutions at each generation

%------------------------------- Reference --------------------------------
% T. Chugh, Y. Jin, K. Miettinen, J. Hakanen, and K. Sindhya, A surrogate-
% assisted reference vector guided evolutionary algorithm for
% computationally expensive many-objective optimization, IEEE Transactions
% on Evolutionary Computation, 2018, 22(1): 129-142.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He

properties (SetAccess = private)
    alpha = 2;
    wmax = 20;
    mu = 5;
    MuC = 20;
    MuM = 15;
end

methods
    function Parameter = getParameter(Algo)
        Parameter = {'alpha: ', num2str(Algo.alpha), ...
            'wmax: ', num2str(Algo.wmax), ...
            'mu: ', num2str(Algo.mu), 'MuC: Simulated Binary Crossover', num2str(Algo.MuC), 'MuM: Polynomial Mutation', num2str(Algo.MuM)};
    end
    
    function setParameter(Algo, Parameter)
        i = 1;
        Algo.alpha = str2double(Parameter{i}); i = i + 1;
        Algo.wmax = str2double(Parameter{i}); i = i + 1;
        Algo.mu = str2double(Parameter{i}); i = i + 1;
        Algo.MuC = str2double(Parameter{i}); i = i + 1;
        Algo.MuM = str2double(Parameter{i}); i = i + 1;
    end
    
    function run(Algo, Prob)
        warning off;
        % Initialize
        for t = 1:Prob.T
            [V0{t}, N{t}] = UniformPoint(Prob.N, Prob.M(t));
            V{t} = V0{t};
%             NI    = 11*Prob.D{t}-1;
            NI    = 100;
            P{t} = UniformPoint(NI,max( Prob.D ) ,'Latin');
%             P{t} =  Prob.Go{t}.problem('init', Prob.Go{t},NI);
            %             population{t} = Initialization_One(Algo, Prob, t, Individual, N{t});
            A2{t} = Algo.Evaluation1(P{t}, Prob, t, Individual);
%             A2{t} = Algo.Evaluation1_Surrogate(P{t}, Prob, t, Individual);
            FA2{t}= A2{t};
            TransferNum{t} = [];
            A1{t} = A2{t};
            THETA{t} = 5.*ones(Prob.M(t),Prob.D(t));
            Model{t} = cell(1,Prob.M(t));
            infill_x{t} = zeros(Algo.mu, max(Prob.D) );
            infill_y{t} = zeros(Algo.mu, Prob.M(t));
            % New
            Ar{t} = [];
            CM = [inf,inf];
        end
        
        while Algo.notTerminated(Prob, A2)
            for t = 1:Prob.T
                disp([Algo.Name, ' Task' num2str(t) ': ' num2str(length(A2{t}))] );
                A1Dec = A1{t}.Decs;
                A1Obj = A1{t}.Objs;
                for i = 1 : Prob.M(t)
                    % The parameter 'regpoly1' refers to one-order polynomial
                    % function, and 'regpoly0' refers to constant function. The
                    % former function has better fitting performance but lower
                    % efficiency than the latter one
%                     dmodel     = dacefit(A1Dec(:, 1:Prob.D(t)),A1Obj(:,i),'regpoly1','corrgauss',THETA{t}(i,:),1e-5.*ones(1,Prob.D(t)),100.*ones(1,Prob.D(t)));
%                     Model{t}{i}   = dmodel;
%                     THETA{t}(i,:) = dmodel.theta;
                    
                    NP = length(A2{t}); %by lay
                    if NP == size(P{t},1) && size(infill_x{t},1) ~= 0
                        dmodel = kriging_theta_train(A1Dec(:, 1:Prob.D(t)),A1Obj(:,i),Prob.Lb{t},Prob.Ub{t},THETA{t}(i,:),1e-5.*ones(1,Prob.D(t)),100.*ones(1,Prob.D(t)));
                        Model{t}{i} = dmodel;
                    elseif NP ~= size(P{t},1) && size(infill_x{t},1) ~= 0
                        model = Model{t}{i};
                        sample_x = model.sx; sample_y = model.sy;
                        dmodel = kriging_incremental(model,infill_x{t}(:, 1:Prob.D(t)),infill_y{t}(:,i),sample_x,sample_y,Prob.Lb{t},Prob.Ub{t});
                        Model{t}{i} = dmodel;
                    end
                end
                
                % Transfer
                CurrentDec = A1Dec;
                if t == 1, st = 2; else, st = 1; end
%                 TransferDec = A1{st}.Decs;
                TransferDec = FA2{st}.Decs;
                
                TransferNum{t} = [TransferNum{t}; size(TransferDec,1)];
%                 dv = abs( size(TransferDec,2)-size(CurrentDec,2) );
%                 if size(TransferDec,2) >= size(CurrentDec,2); TransferDec=TransferDec(:,1:size(CurrentDec,2));  else; TransferDec(:,end+1:end+dv) = 0; end
                PopDec = [CurrentDec; TransferDec]; 
                w      = 1;
                while w <= Algo.wmax
                    drawnow('limitrate');
                    % Generation
                    mating_pool = TournamentSelection(2, size(PopDec,1), 1:size(PopDec,1));
                    OffDec = Algo.Generation(PopDec(mating_pool,:));
                    PopDec = [PopDec;OffDec];
                    % Evaluation
                    [N2,~]  = size(PopDec);
                    PopObj = zeros(N2,Prob.M(t));
                    MSE    = zeros(N2,Prob.M(t));
%                     for i = 1: N2
%                         for j = 1 : Prob.M(t)
%                             [PopObj(i,j),~,MSE(i,j)] = predictor(PopDec(i,1:Prob.D(t)),Model{t}{j});
%                         end
%                     end
                    for j = 1 : Prob.M(t)
                        [PopObj(:,j),MSE(:,j)] = kriging_predictor(PopDec(:, 1:Prob.D(t)),Model{t}{j});
                    end
                    index  = KEnvironmentalSelection(PopObj,V{t},(w/Algo.wmax)^Algo.alpha);
                    PopDec = PopDec(index,:);
                    PopObj = PopObj(index,:); 
                    % Adapt referece vectors
                    if ~mod(w,ceil(Algo.wmax*0.1))
                        V{t}(1:N{t},:) = V0{t}.*repmat(max(PopObj,[],1)-min(PopObj,[],1),size(V0{t},1),1);
                    end
                    w = w + 1;
                end
                
                % Select mu solutions for re-evaluation
                infill_x{t} = [];  infill_y{t} = [];
                [NumVf,~] = NoActive(A1Obj,V0{t});
                PopNew    = KrigingSelect(PopDec,PopObj,MSE(index,:),V{t},V0{t},NumVf,0.05*N{t},Algo.mu,(w/Algo.wmax)^Algo.alpha);
                New       = Algo.Evaluation1(PopNew, Prob, t, Individual); %%????
                A2{t}        = [A2{t},New];
                A1{t}        = UpdataArchive(A1{t},New,V{t},Algo.mu,NI);
                infill_x{t} = New.Decs; infill_y{t} = New.Objs;
                [FrontNo,~] = NDSort(A2{t}.Objs,1);
                FA2{t} = A2{t}(FrontNo == 1);
            end
        end
    end
    
    function OffDec = Generation(Algo, PopDec)
        count = 1;
        for i = 1:ceil(size(PopDec,1) / 2)
            p1 = i; p2 = i + fix(size(PopDec,1) / 2);
            PopDec(count,:) = PopDec(p1,:);
            PopDec(count + 1,:) = PopDec(p2,:);

            [OffDec(count,:), OffDec(count + 1,:)] = GA_Crossover(PopDec(p1,:), PopDec(p2,:), Algo.MuC);

            OffDec(count,:) = GA_Mutation(OffDec(count,:), Algo.MuM);
            OffDec(count + 1,:) = GA_Mutation(OffDec(count + 1,:), Algo.MuM);

            for x = count:count + 1
                Dec = OffDec(x,:);
%                 if (sum(Dec > 1) + sum(Dec < 0) > 0)
%                     exception = sum(Dec > 1) + sum(Dec < 0);
%                 end
                OffDec(x, Dec > 1) = 1;
                OffDec(x, Dec < 0) = 0;
%                 OffDec(x).Dec(OffDec(x).Dec > 1) = 1;
%                 OffDec(x).Dec(OffDec(x).Dec < 0) = 0;
            end
            count = count + 2;
        end
    end
    
end
end
