% -------------------------------------------------------------------------
% tabula rasa
% -------------------------------------------------------------------------
     clear all
     close all
     clc

%% 
% -------------------------------------------------------------------------
% import data
% -------------------------------------------------------------------------
[FName,PathName]=uigetfile('*.txt','Select Text file');
sample=importdata([PathName,FName]);
%
time_301_b=sample.data(:,1);cf tool


for i=2:length(time_301_b);
    if time_301_b(i)-time_301_b(i-1)==10;
        time_301_b(i)=time_301_b(i)-5;      
    else 
        time_301_b(i)=time_301_b(i);        
    end
end
        
Dat_301_b=sample.data(:,2:9);

%%
% -------------------------------------------------------------------------
% transform MEO data
% -------------------------------------------------------------------------
Dat_301_b(:,1)=Dat_301_b(:,1);
Dat_301_b(:,2)=Dat_301_b(:,2);
Dat_301_b(:,3)=Dat_301_b(:,3);
Dat_301_b(:,4)=Dat_301_b(:,4);
Dat_301_b(:,5)=Dat_301_b(:,5);
Dat_301_b(:,6)=Dat_301_b(:,6);
Dat_301_b(:,7)=Dat_301_b(:,7);
Dat_301_b(:,8)=Dat_301_b(:,8);
%% save raw data in figure
figure('units','normalized','outerposition',[0 0 1 1])
for l=1:8;
subplot(2,4,l)
hold on
plot(time_301_b/60,Dat_301_b(:,l)*10^6,'linewidth',2)
ylabel('Current (\muA)')
xlabel('Time (min)')
box on
end

h=gcf;
set(findall(h,'-property','FontSize'),'LineWidth',2)
set(h,'PaperOrientation','landscape');
set(h,'PaperUnits','normalized');
set(h,'PaperPosition', [0 0 1 1]);

%,'FontWeight','bold'
%print(gcf,'-dpdf','fig_2')
        
set(gcf,'color','w');

saveas(gcf,'Fig_raw_301_b.pdf')
%% 
% -------------------------------------------------------------------------
% baseline subtraction
% -------------------------------------------------------------------------
% "current" is plotted and the user is instructed to select points for the 
% fit. A baseline will be linearly interpolated from the selected points 
% and will be plotted together with "y". The user is prompted as to whether 
% to redo the baseline selection. Upon completion, the corrected data 
% "current" and the fitted baseline "bs" are output.
[Dat_301_b(:,1),bs(:,1)] = bf(Dat_301_b(:,1),'confirm',16); 
[Dat_301_b(:,2),bs(:,2)] = bf(Dat_301_b(:,2),'confirm',16); 
[Dat_301_b(:,3),bs(:,3)] = bf(Dat_301_b(:,3),'confirm',16); 
[Dat_301_b(:,4),bs(:,4)] = bf(Dat_301_b(:,4),'confirm',16); 
[Dat_301_b(:,5),bs(:,5)] = bf(Dat_301_b(:,5),'confirm',16); 
[Dat_301_b(:,6),bs(:,6)] = bf(Dat_301_b(:,6),'confirm',16); 
[Dat_301_b(:,7),bs(:,7)] = bf(Dat_301_b(:,7),'confirm',16); 
[Dat_301_b(:,8),bs(:,8)] = bf(Dat_301_b(:,8),'confirm',16);
%%
% -------------------------------------------------------------------------
% save baseline subtraction
% -------------------------------------------------------------------------
% Save Output
csvwrite('bs_301_b.csv',Dat_301_b)
%%
% -------------------------------------------------------------------------
% find peak start and end point
% -----------------------------------------------------------------------
msviewer(time_301_b,Dat_301_b(:,1))
% open desired current (:,x) in the ms viewer and chose the start and end % points of the peak by right clicking on the data, moving 
% line to desired position and then export markers to workspace as
% c1, c2, etc.
%% 
% -------------------------------------------------------------------------
% User Input Section
% -------------------------------------------------------------------------
% NPeaks=[2,2,2,2,2,2,2,2]; % 
 NPeaks=[1,1,1,1,1,1,1,1]; % _301_b
% NPeaks=[3,3,3,3,3,3,3,3]; % _301_b
% NPeaks=[4,4,4,4,4,4,4,4]; % _301_b
% NPeaks=[4,4,4,4,4,4,4,4]; % _301_b
% NPeaks=[5,5,5,5,5,5,5,5]; % _301_b
% sep=[c1,c2,c3,c4,c5,c6,c7,c8];
% sep_start=sep([1:2:8],:); % select all peak start points
% sep_start=round(sep_start./5)*5/5; % round to 5 (as dt=5)
% sep_end=sep([2:2:9],:); % select all peak end points
% NPeaks=[5,5,5,5,5,5,5,5]; % _301_b
sep=[c1,c2,c3,c4,c5,c6,c7,c8];
sep_start=sep([1:2:2],:); % select all peak start points
sep_start=round(sep_start./5)*5/5; % round to 5 (as dt=5)
sep_end=sep([2:2:2],:); % select all peak end points
sep_end=round(sep_end./5)*5/5; % round to 5 (as dt=5)

% sep_start=sep([1:2:3],:); % select all peak start points
% sep_start=round(sep_start./5)*5/5; % round to 5 (as dt=5)
% sep_end=sep([2:2:4],:); % select all peak end points
% sep_end=round(sep_end./5)*5/5; % round to 5 (as dt=5)

% sep_start=sep([1:2:5],:); % select all peak start points
% sep_start=round(sep_start./5)*5/5; % round to 5 (as dt=5)
% sep_end=sep([2:2:6],:); % select all peak end points
% sep_end=round(sep_end./5)*5/5; % round to 5 (as dt=5)
% cmd+f to replace all experiment numbers _xxx
% -------------------------------------------------------------------------
% Parameters and matrixes
% -------------------------------------------------------------------------
f=96485.3365;
dt=5;
Out=zeros(sum(NPeaks),11);
%RawDatOut_301_b=zeros(sum(NPeaks),10000)*NaN;
%% 
% -------------------------------------------------------------------------
% Loop over currents and peaks
% -------------------------------------------------------------------------
for i=1:length(Dat_301_b(1,:))
    % for currents 1 to 8
    
    for j=1:NPeaks(i)
        % for all peaks
        
        % Find start and end index
%         i1=Xdat(ExStartLine-1+sum(NPeaks(1:(i-1)))+j-ExHeaderLines,25);
%         i1=i1+1; % Unterschied Index Igor vs. Matlab 
%         i2=Xdat(ExStartLine-1+sum(NPeaks(1:(i-1)))+j-ExHeaderLines,26);
%         i2=i2+1;
        i1=sep_start(j,i);
        i2=sep_end(j,i);
        
        % Find I max and index of I max
        clear ind_imax
        [~,ind_imax]=max(Dat_301_b(i1:i2,i));
        %[~,ind_imax]=max(Dat(i1+50:i2,i+1));
        clear imax
        imax=max(Dat_301_b(i1:i2,i));
        Out(sum(NPeaks(1:(i-1)))+j,1)=imax;
        
        % Smooth I
        sI=Dat_301_b(i1:i2,i);
        % I=Dat(i1:i2,i);
        % sI=smooth(I);
        % ssI=smooth(sI);
        
        % Integrate I, divide by f, subtract int from int(end) to get mole
        % of Fe(III) left
        clear int
        int=zeros(1,length(sI));
        int(1)=0;
        for k=2:length(sI)
            int(k)=int(k-1)+((sI(k)+sI(k-1))/2)*dt;
        end
        int=int/f;
        Out(sum(NPeaks(1:(i-1)))+j,2)=int(end);
        int=int(end)-int;
        %---
        sI_Imax=Dat_301_b(i1:i1+ind_imax,i);
        clear int_Imax
        int_Imax=zeros(1,length(sI_Imax));
        int_Imax(1)=0;
        for l=2:length(sI_Imax)
            int_Imax(l)=int_Imax(l-1)+((sI_Imax(l)+sI_Imax(l-1))/2)*dt;
        end
        int_Imax=int_Imax/f;
        Out(sum(NPeaks(1:(i-1)))+j,11)=int_Imax(end);
        int_Imax=int_Imax(end)-int_Imax;
        %-----
        % Find ind_95%, i.e. time value when 95% of initial Fe(III) is
        % reduced
        clear ind_95
        val05=0.05*int(1);
        [~,ind_95]=min(abs(int-val05));
        % subtract val95 in order to find t value corresponding to 95% Int
        
 %       Out(sum(NPeaks(1:(i-1)))+j,11)=[int(ind_imax)];        
        
        % Find ind_50%
        clear ind_50
        val5=0.5*int(1);
        [~,ind_50]=min(abs(int-val5));
        
        % Find ind_20%
        clear ind_20
        val5=0.8*int(1);
        [~,ind_20]=min(abs(int-val5));
        
        % Find ind_50%
        clear ind_80
        val5=0.2*int(1);
        [~,ind_80]=min(abs(int-val5));
        
        % Determine Fe(III)
        clear Fe3
        Fe3_301_b = int/int(1);

% % -------------------------------------------------------------------------
% % Fitting
% % -------------------------------------------------------------------------        
% % fit linear to log Fe3 from ind_301_bx to ind_95
%         m1x=time_301_b(ind_imax:ind_95).';
%         m1y=log(Fe3_301_b(ind_imax:ind_95));
%         [xm1x, ym1y] = prepareCurveData( m1x, m1y );
% 
%         % Set up fittype and options.
%         ft = fittype( 'm*x+c', 'independent', 'x', 'dependent', 'y' );
%         opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
%         opts.Display = 'Off';
%         opts.StartPoint = [0.847139088616202 0.546881519204984];
% 
%         % Fit model to data.
%         [m1fitresult, m1gof] = fit( xm1x, ym1y, ft, opts );
%         
%         % Coefficients       
%         m1coeff=coeffvalues(m1fitresult);
%         m1a=2.71828182846^(m1coeff(1));
%         m1b=m1coeff(2);
%         
%         % Save fit to Out
%         Out(sum(NPeaks(1:(i-1)))+j,3:6)=[m1b,m1a,m1gof.rsquare,...
%             m1gof.rmse];
% % ------------------------------------------------------------------------- 
% % fit linear to log Fe3 from ind_50 to ind_95
%         m2x=time_301_b(ind_50:ind_80).';
%         m2y=log(Fe3_301_b(ind_50:ind_80));
%         [xm2x, ym2y] = prepareCurveData( m2x, m2y );
% 
%         % Set up fittype and options.
%         ft = fittype( 'm*x+c', 'independent', 'x', 'dependent', 'y' );
%         opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
%         opts.Display = 'Off';
%         opts.StartPoint = [0.847139088616202 0.546881519204984];
% 
%         % Fit model to data.
%         [m2fitresult, m2gof] = fit( xm2x, ym2y, ft, opts );
%         
%         % Coefficients       
%         m2coeff=coeffvalues(m2fitresult);
%         m2a=2.71828182846^(m2coeff(1));
%         m2b=m2coeff(2);
%         
%         % Save fit to Out
%         Out(sum(NPeaks(1:(i-1)))+j,7:10)=[m2b,m2a,m2gof.rsquare,...
%             m2gof.rmse];

% -------------------------------------------------------------------------
% Save data
% -------------------------------------------------------------------------
        % Save raw data in RawDatOut
        DatSel=Dat_301_b(i1:i2,i);
        RawDatOut_301_b(sum(NPeaks(1:(i-1)))+j,1:i2-i1+1)=[DatSel];
% -------------------------------------------------------------------------
% Plot data
% -------------------------------------------------------------------------
%  %      Prepare Plots
%         if i1<21
%             istart=1;
%         else
%             istart=i1-20;
%         end
%         figure('units','normalized','outerposition',[0 0 1 1])
%         
%  %      plot I vs t
%         subplot(1,3,1)
%         hold on
%         plot(time_301_b(istart:i2),Dat_301_b(istart:i2,i))
%         ymin=min(Dat_301_b(i1:i2,i));
%         ymax=max(Dat_301_b(i1:i2,i));
%         ymin=ymin-(ymax-ymin)*0.05;
%         ymax=ymax+(ymax-ymin)*0.05;
%         plot(ones(2,1)*time_301_b(i1),[ymax,ymin],'k')
%         plot(ones(2,1)*time_301_b(i2),[ymax,ymin],'k')      
%         plot(ones(2,1)*time_301_b(i1+ind_imax),[ymax,ymin],'c')
%         plot(ones(2,1)*time_301_b(i1+ind_95),[ymax,ymin],'g')
%         plot(ones(2,1)*time_301_b(i1+ind_50),[ymax,ymin],'b')
%         xlabel('Time t [s]')
%         ylabel('Current I [A]')
%         title(['Dataset ',num2str(i),'.',num2str(j),' - I vs. t'])
%         box on
%                 
% %       plot m/m0 vs t incl. fit reactivity distribution
%         subplot(1,3,2)
%         hold on
%         h = plot( m1fitresult, m1x, m1y ,'k');
%         legend off
%         plot(ones(2,1)*time_301_b(ind_imax),[min(m1y),max(m1y)],'c');
%         plot(ones(2,1)*time_301_b(ind_95),[min(m1y),max(m1y)],'g');
%         plot(ones(2,1)*time_301_b(ind_50),[min(m1y),max(m1y)],'b');
%         txt={'y=m*x+c from imax to 95%','m = ',num2str(m1b),'c =',num2str(m1a),...
%             ' R^2 = ',num2str(m1gof.rsquare),...
%             ' RMSE = ',num2str(m1gof.rmse)};
%         text(0.3,0.2,txt,'color','k','units','normalized',...
%         'HorizontalAlignment','left','VerticalAlignment','middle')
%         xlabel('Time t [s]')
%         ylabel('log(Fe(III)) [mol]')
%         title(['Dataset ',num2str(i),'.',num2str(j),' - m/m0 vs. t'])
%         box on
%         
% %       plot m/m0 vs t incl. fit reactivity distribution
%         subplot(1,3,3)
%         hold on
%         h = plot( m2fitresult, m2x, m2y ,'k');
%         legend off
%         plot(ones(2,1)*time_301_b(ind_imax),[min(m2y),max(m2y)],'c');
%         plot(ones(2,1)*time_301_b(ind_50),[min(m2y),max(m2y)],'b');
%         plot(ones(2,1)*time_301_b(ind_95),[min(m2y),max(m2y)],'g');
%         txt={'y=m*x+c from 50-95%','m = ',num2str(m2b),'c =',num2str(m2a),...
%             ' R^2 = ',num2str(m2gof.rsquare),...
%             ' RMSE = ',num2str(m2gof.rmse)};
%         text(0.3,0.2,txt,'color','k','units','normalized',...
%         'HorizontalAlignment','left','VerticalAlignment','middle')
%         xlabel('Time t [s]')
%         ylabel('log(Fe(III)) [mol]')
%         title(['Dataset ',num2str(i),'.',num2str(j),' - m/m0 vs. t'])
%         box on     
%         
%         Nfig=1000+ceil(((i-1)*max(NPeaks)+j)/8);
%         if((mod((i-1)*max(NPeaks)+j,8))==0)
%             Nindex=8;
%         else
%             Nindex=mod((i-1)*max(NPeaks)+j,8);
%         end
     
        
%         Nfig=1000+ceil(((i-1)*max(NPeaks)+j)/6);
%         if((mod((i-1)*max(NPeaks)+j,6))==0)
%             Nindex=6;
%         else
%             Nindex=mod((i-1)*max(NPeaks)+j,6);
%         end
%         
%         figure(Nfig)
%         set(Nfig,'units','normalized','outerposition',[0 0 1 1])
%         % plot m/m0 vs t incl. fit reactivity distribution
%         subplot(2,max(NPeaks),Nindex)
%         hold on
%         h = plot( m1fitresult, m1x, m1y ,'k');
%         legend off
%         plot(ones(2,1)*time_301_b(ind_imax),[min(m1y),max(m1y)],'c');
%         plot(ones(2,1)*time_301_b(ind_95),[min(m1y),max(m1y)],'g');
%         plot(ones(2,1)*time_301_b(ind_50),[min(m1y),max(m1y)],'b');
%         txt={'y=m*x+c from imax to 95%','m = ',num2str(m1b),'c =',num2str(m1a),...
%             ' R^2 = ',num2str(m1gof.rsquare),...
%             ' RMSE = ',num2str(m1gof.rmse)};
%         text(0.3,0.2,txt,'color','k','units','normalized',...
%         'HorizontalAlignment','left','VerticalAlignment','middle')
%         xlabel('Time t [s]')
%         ylabel('log(Fe(III)) [mol]')
%         title(['Dataset ',num2str(i),'.',num2str(j),' - m/m0 vs. t'])
%         set(gcf,'color','w');
%         box on        
        
    end       
end

% %% Export figs
% export_fig(1001,'fits_301_b_c1_c2.pdf')
% export_fig(1002,'fits_301_b_c3_c4.pdf')
% export_fig(1003,'fits_301_b_c5_c6.pdf')
% export_fig(1004,'fits_301_b_c7_c8.pdf')
%%
close all
%% Save data and write Output
RawDatOut_301_b(sum(NPeaks(1:(i-1)))+j+1,1:length(time_301_b))=[time_301_b];
csvwrite('RawDatOut_301_b.csv',RawDatOut_301_b)
% Save Output
csvwrite('Output_301_b.csv',Out)

% Save Workspace
save('WS_301_b')
%% Save plot of data
figure('units','normalized','outerposition',[0 0 1 1])
for l=1:8;
subplot(2,4,l)
hold on
% plot(time_301_b/60,Dat_301_b(:,l)*10^6,'linewidth',2,'color','red')
plot(time_301_b/60,Dat_301_b(:,l)*10^6,'linewidth',2,'color','k')

ylabel('Current (\muA)')
xlabel('Time (min)')
box on
end

h=gcf;
set(findall(h,'-property','FontSize'),'LineWidth',2)
set(h,'PaperOrientation','landscape');
set(h,'PaperUnits','normalized');
set(h,'PaperPosition', [0 0 1 1]);

%,'FontWeight','bold'
%print(gcf,'-dpdf','fig_2')
        
set(gcf,'color','w');

saveas(gcf,'Fig_301_b.pdf')