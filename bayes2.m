function latents = bayes2(data, param)

    % smarter Bayesian learning (assumes one reward per state)
    % learn posterior over P(r|s,a=1) for each s, with Beta prior
    % with optional decay of Beta parameters
    % reset Beta parameters for each session
    %

    if ~exist('param', 'var')
        alpha0 = 1; % beta param
        beta0 = 1; % beta param
        tau = 1; % softmax temperature
        decay = 0.1; % decay rate
    else
        alpha0 = param(1);
        beta0 = param(2);
        tau = param(3);
        decay = param(4);
    end

    S = max(data.cue); % # states / cues
    A = 2; % # actions

    for i = 1:length(data.cue)
        s = data.cue(i);

        if i == 1 || (i > 1 && data.sesh(i - 1) ~= data.sesh(i))
            % reset for each session
            alpha = ones(1,S) * alpha0;
            beta = ones(1,S) * beta0;
        end

        [P, ~] = betastat(alpha, beta); % P(r|s,a=1) = mean of beta for s
        Q(:,1) = P'; % Q(s,a=1) = P(r|s,a=1)
        Q(:,2) = 1 - P'; % Q(s,a=2) = P(r|s,a=2) = 1 - P(r|s,a=1)

        p = softmax(Q(s,:), tau);
        a = find(mnrnd(1, p));

        reward = data.r(i,a);
        PE = reward - Q(s,a);

        alpha = alpha * (1 - decay) + alpha0 * decay;
        beta = beta * (1 - decay) + beta0 * decay;
        if (reward == 1 && a == 1) || (reward == 0 && a == 2)
            alpha(s) = alpha(s) + 1; % a = 1 was rewarding
        else
            beta(s) = beta(s) + 1; % a = 2 was rewarding
        end

        latents.allQ(:,:,i) = Q;
        latents.alpha(i,:) = alpha;
        latents.beta(i,:) = beta';
        latents.Q(i,:) = Q(s,:);
        latents.reward(i) = reward;
        latents.PE(i) = PE;
        latents.p(i,:) = p;
        latents.a(i) = a;
    end

end

function p = softmax(Q, tau)
    p = exp(Q / tau);
    p = p / sum(p);
end




