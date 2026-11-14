function x = fixes(x, which)
% Minimal AC fixes. 'J': share zone-J generators between NYC buses 81 and 82 in proportion
% to their load (NYgrid puts all at 82, none at 81). 'L': restore the 765 kV Massena-Marcy line
% that NYgrid's modifyMPC.m carries commented out (47-43, r 0.004, x 0.0085).
define_constants;
if contains(which,'J')
  g = find(x.gen(:,GEN_BUS)==82); l81 = x.bus(x.bus(:,BUS_I)==81,PD); l82 = x.bus(x.bus(:,BUS_I)==82,PD);
  n = round(numel(g) * l81/(l81+l82)); x.gen(g(1:2:2*n),GEN_BUS) = 81;
  x.bus(x.bus(:,BUS_I)==81,BUS_TYPE) = PV;
end
if contains(which,'L')
  nb = zeros(1,size(x.branch,2)); nb(1:13) = [47 43 0.004 0.0085 0 0 0 0 0 0 1 -360 360]; x.branch = [x.branch; nb];
end
end
