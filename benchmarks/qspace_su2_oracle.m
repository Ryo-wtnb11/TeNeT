% QSpace SU(2) spin-half oracle for TeNeT issue #9.
%
% Reference environment:
%   QSpace public master e87ccd14d2efdfd50b66fb58e85ac1d1b3000347
%   MATLAB R2026a Update 4, Xcode 16.4, Apple Clang 17.0.0
%
% Run after QSpace startup with getSymStates, compactQS, contractQS, plusQS,
% uniquerows, and eigQS built for mexmaca64.

[S, IS] = getLocalSpace('Spin', 1 / 2);
labels = cellfun(@(q) q(1), S.Q);
assert(isequal(labels, [1, 1, 2]));
assert(isequal(S.info.itags, {'', '*', '*'}));
assert(abs(S.data{1} + sqrt(3) / 2) <= 1e-14);

S2 = contract(S, '13*', S, '13');
assert(rank(S2) == 2);
assert(numel(S2.data) == 1);
assert(abs(S2.data{1} - 3 / 4) <= 1e-14);
assert(abs(IS.E.data{1} - 1) <= 1e-14);

fprintf('== QSpace SU2 spin-half oracle ==\n');
fprintf('labels = [%d, %d, %d]\n', labels);
fprintf('itags = ["", "*", "*"]\n');
fprintf('reduced = %.15e\n', S.data{1});
fprintf('casimir = %.15e\n', S2.data{1});
fprintf('identity = %.15e\n', IS.E.data{1});
fprintf('closed_norm2 = %.15e\n', 2 * S2.data{1});
