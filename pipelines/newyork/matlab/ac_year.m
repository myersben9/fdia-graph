function ac_year(mpdir, hours, outfile)
% Year-long AC feasibility of NYgrid with the minimal recipe:
%   raw     : NYgrid 2019 operating point, full 140-bus case, plain AC PF
%   recipe  : fixes 'JL' (zone-J gens split 81/82, 765 kV Massena-Marcy line) + external
%             areas at base load (extbase), then the min-deviation AC-OPF projection (acproj)
% Interface flows use zone cuts on the full network, so DC and AC are compared on the same
% definitions against NYISO's historical interface flows.
addpath(genpath(mpdir)); addpath(genpath(pwd)); define_constants;
o = mpoption('verbose',0,'out.all',0,'exp.use_legacy_core',1);
mpc0 = modifyMPC(); m1 = mpc0;
T = readtable(fullfile('Data','npcc.csv'),'TextType','string'); zmap = containers.Map(num2cell(T.idx), cellstr(T.zone));
cuts = {{'DYSINGER EAST','A','B'},{'WEST CENTRAL','B','C'},{'MOSES SOUTH','D','E'},{'CENTRAL EAST - VC','E','FG'},{'TOTAL EAST','ABCDE','FGHIJK'},{'UPNY CONED','G','HIJ'},{'SPR/DUN-SOUTH','I','J'}};
fid = fopen(outfile,'w');
fprintf(fid,'ts,nyload,raw_ac_ok,dc_dthmax,fix_plain_ok,proj_ok,redisp_ny,ny_gen,redisp_ext,vmin,vmax,ny_vmin,ny_vmax,ac_dthmax,loss_ac,red_dc_ifmae,dc_ifmae,ac_ifmae,acdc_maxdiff\n');
for k = 1:numel(hours)
  ts = hours(k);
  try
    [mr,mf] = updateOpCond(mpc0,ts,false,false,true);
    [~,interFlow,flowLimit,~,~,~] = readOpCond(ts);
    raw = runpf(mf,o); dcr = rundcpf(mr,o); [~,~,er] = flow4Plot(dcr,interFlow,flowLimit);
    x = extbase(fixes(mf,'JL'), m1); d = rundcpf(x,o); p = runpf(x,o);
    [rp,info] = acproj(x,o);
    nyb = x.bus(ismember(x.bus(:,BUS_AREA),[2 3]),BUS_I); nyg = ismember(x.gen(:,GEN_BUS),nyb); ny = ismember(x.bus(:,BUS_I),nyb);
    edc = iferr(d,cuts,zmap,interFlow,flowLimit);
    if rp.success
      dp = rp.gen(:,PG)-x.gen(:,PG); eac = iferr(rp,cuts,zmap,interFlow,flowLimit);
      fac = ifflow(rp,cuts,zmap); fdc = ifflow(d,cuts,zmap);
      vals = [sum(abs(dp(nyg)))/2, sum(x.gen(nyg,PG)), sum(abs(dp(~nyg)))/2, min(rp.bus(:,VM)), max(rp.bus(:,VM)), min(rp.bus(ny,VM)), max(rp.bus(ny,VM)), dthmax(rp), sum(real(get_losses(rp))), mean(abs(er)), mean(abs(edc)), mean(abs(eac)), max(abs(fac-fdc))];
    else
      vals = [nan(1,9), mean(abs(er)), mean(abs(edc)), nan, nan];
    end
    fprintf(fid,'%s,%.1f,%d,%.1f,%d,%d%s\n', datestr(ts,'yyyy-mm-dd HH'), sum(mf.bus(ny,PD)), raw.success, dthmax(d), p.success, rp.success, sprintf(',%.4f',vals));
    if k <= 5 || mod(k,200)==0, fprintf('%s raw=%d fixplain=%d proj=%d redispNY=%.0f V %.3f-%.3f ifMAE redDC %.3f DC %.3f AC %.3f\n', datestr(ts,'yyyy-mm-dd HH'), raw.success, p.success, rp.success, vals(1), vals(4), vals(5), vals(10), vals(11), vals(12)); end
  catch e
    fprintf(fid,'%s,ERR %s\n', datestr(ts,'yyyy-mm-dd HH'), strrep(e.message,',',';'));
  end
end
fclose(fid);
end
function f = ifflow(r,cuts,zmap)
  define_constants; f = zeros(numel(cuts),1);
  zf = zz(r.branch(:,F_BUS),zmap); zt = zz(r.branch(:,T_BUS),zmap);
  for c = 1:numel(cuts)
    a = cuts{c}{2}; b = cuts{c}{3};
    fw = ismember(zf,num2cell(a)) & ismember(zt,num2cell(b)); bw = ismember(zt,num2cell(a)) & ismember(zf,num2cell(b));
    f(c) = sum(r.branch(fw,PF)) - sum(r.branch(bw,PF));
  end
end
function e = iferr(r,cuts,zmap,interFlow,flowLimit)
  f = ifflow(r,cuts,zmap); e = nan(numel(cuts),1);
  for c = 1:numel(cuts)
    h = interFlow.FlowMWH(interFlow.InterfaceName == cuts{c}{1}); L = flowLimit(flowLimit.InterfaceName == cuts{c}{1},:);
    if ~isempty(h) && ~isempty(L), e(c) = (h(1) - f(c))/L.PositiveLimitMWH(1); end
  end
end
function z = zz(b,zmap)
  z = cell(numel(b),1); for i = 1:numel(b), if isKey(zmap,b(i)), z{i} = zmap(b(i)); else, z{i} = 'X'; end, end
  z = cellfun(@(s) s(1), z, 'UniformOutput', false);
end
function v = dthmax(r)
  define_constants; [~,fi] = ismember(r.branch(:,F_BUS), r.bus(:,BUS_I)); [~,ti] = ismember(r.branch(:,T_BUS), r.bus(:,BUS_I));
  on = r.branch(:,BR_STATUS)>0; v = max(abs(r.bus(fi(on),VA)-r.bus(ti(on),VA)));
end
