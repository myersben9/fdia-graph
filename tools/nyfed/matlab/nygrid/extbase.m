function x = extbase(x, m1)
% Keep external areas at base-case load; scale external generation to cover NY's net import.
define_constants;
nyb = x.bus(ismember(x.bus(:,BUS_AREA),[2 3]),BUS_I); ext = ~ismember(x.bus(:,BUS_I),nyb);
x.bus(ext,[PD QD]) = m1.bus(ext,[PD QD]); ge = ~ismember(x.gen(:,GEN_BUS),nyb) & x.gen(:,GEN_STATUS)>0;
need = sum(x.bus(:,PD)) - sum(x.gen(~ge & x.gen(:,GEN_STATUS)>0,PG)); x.gen(ge,PG) = x.gen(ge,PG)*need/sum(x.gen(ge,PG));
end
