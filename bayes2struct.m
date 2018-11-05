function latents = bayes2struct(data, param)


    if ~exist('param', 'var')
        alpha0 = 1; % beta param
        beta0 = 1; % beta param
        tau = 1; % softmax temperature
        decay = 0; % decay rate
        alpha = 5; % sCRP concentration parameter
        lambda = 0.5; % sCRP stickiness parameter
    else
        alpha0 = param(1);
        beta0 = param(2);
        tau = param(3);
        decay = param(4);
        alpha = param(5);
        lambda = param(6);
    end

    S = max(data.cue); % # states / cues
    A = 2; % # actions
    K = 20; % max # event types
    maxh = 100; % max # of particles

    h.e = []; % event history
    h.C = zeros(1,K); % event type counts
    h.alpha = ones(K,S) * alpha0; % beta param
    h.beta = ones(K,S) * beta0; % beta param
    h.w = 1; % importance weight

    particles = [h];

    for i = 1:length(data.cue)
        s = data.cue(i);
        i
        save shit.mat;

        % prior
        particles = fork(particles, alpha, lambda, K);

        % Q-values
        Q = computeQ(particles, S);

        % action selection
        p = softmax(Q(s,:), tau);
        a = find(mnrnd(1, p));

        % reward
        reward = data.r(i,a);
        PE = reward - Q(s,a);

        % decay TODO remove?
        for j = 1:length(particles)
            particles(j).alpha = particles(j).alpha * (1 - decay) + alpha0 * decay;
            particles(j).beta = particles(j).beta * (1 - decay) + beta0 * decay;
        end

        % update
        if (reward == 1 && a == 1) || (reward == 0 && a == 2)
            % a = 1 was rewarding
            for j = 1:length(particles)
                h = particles(j);
                particles(j).w = betastat(h.alpha(h.e(end),s), h.beta(h.e(end),s)) * h.w;
                particles(j).alpha(h.e(end),s) = particles(j).alpha(h.e(end),s) + 1;
            end
        else
            % a = 2 was rewarding
            for j = 1:length(particles)
                h = particles(j);
                particles(j).w = (1 - betastat(h.alpha(h.e(end),s), h.beta(h.e(end),s))) * h.w;
                particles(j).beta(h.e(end),s) = particles(j).beta(h.e(end),s) + 1;
            end
        end

        % pick top particles, if too many
        if length(particles) > maxh
            [~,I] = sort([particles.w], 'descend');
            particles = particles(I(1:maxh));
        end

        % normalize importance weights
        particles = normalize(particles);

        % logging
        latents.allQ(:,:,i) = Q;
        latents.Q(i,:) = Q(s,:);
        latents.particles{i} = particles;
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

% make importance weights sum to 1
%
function particles = normalize(particles)
    Z = sum([particles.w]);
    for j = 1:length(particles)
        particles(j).w = particles(j).w / Z;
    end
end

% fork each particle into K new particles by adding a new
% event of each type and adjusting importance weight according to sCRP:
%
% P(e_1:i|history) = P(e_i|e_1:i-1) * P(e_1:i-1|history) = (sCRP prior) * (importance weight)
%
function new_particles = fork(particles, alpha, lambda, K)
    new_particles = [];
    for j = 1:length(particles)
        h = particles(j);

        % sCRP prior
        C = h.C;
        maxe = 1;
        if length(h.e) > 0
            maxe = max(h.e) + 1;
            C(h.e(end)) = C(h.e(end)) + lambda;
        end
        C(maxe) = alpha;
        P = C / sum(C); 

        % create K new particles from particle h
        for k = 1:maxe
            h_new = h;
            h_new.e = [h.e k];
            h_new.C(k) = h.C(k) + 1;
            h_new.w = P(k) * h.w;
            new_particles = [new_particles; h_new];
        end
    end

    new_particles = normalize(new_particles);
end

% Compute Q(s,a) = P(r|s,a) by marginalizing over samples
%
% P(r|s,a=1,history) = sum over all e_1:i of P(r|s,a=1,e_1:i) * P(e_1:i|history)
%                   ~= sum over samples of P(r|s,a=1,e_1:i) * (importance weight)
%                    = sum over samples of (beta mean) * (importance weight)
%
function Q = computeQ(particles, S)
    assert(abs(sum([particles.w]) - 1) < 1e-9);

    Q = zeros(S,2);
    for j = 1:length(particles)
        h = particles(j);

        % P(r|s,a=1,e_1:i) * P(e_1:i|history)
        % = (beta mean) * (importance weight)
        P = betastat(h.alpha(h.e(end),:), h.beta(h.e(end),:)) * h.w;

        Q(:,1) = Q(:,1) + P';
    end

    Q(:,2) = 1 - Q(:,1);
end
