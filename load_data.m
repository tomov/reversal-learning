function data = load_data(filename)

T = readtable(filename);

for s = unique(T.subj)'
    data(s).sesh = T.sesh(T.subj == s);
    data(s).block = T.block(T.subj == s);
    data(s).trial = T.trial(T.subj == s);
    data(s).cond = T.cond(T.subj == s);
    data(s).cue = T.cue(T.subj == s);
    data(s).r = [T.r1(T.subj == s) T.r2(T.subj == s)];
    data(s).choice = T.choice(T.subj == s);
    data(s).reward = T.reward(T.subj == s);
end
