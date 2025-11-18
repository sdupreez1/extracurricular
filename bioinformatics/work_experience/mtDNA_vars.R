library(vcfR)
  # to load .vcf file
library(circlize)
library(tidyr)
library(dplyr)
library(stringr)
library(ggplot2)
library(magrittr)

# Data
pgen = read.csv('MITOMAP.csv') %>%
  mutate(Mutation = str_extract(Allele, "[^0-9]*$"),
         Allele = str_extract(Allele, '[0-9].*$')) %>%
  rename(POS = Position) %>%
  mutate(POS = as.numeric(POS))
  # pathogenic variants

npgenic = nrow(pgen)

m.vnts = read.vcfR('mutect2.mutect2.10.forSeth.vcf')
m.vnts = data.frame(m.vnts@fix, m.vnts@gt) %>% 
  filter((FILTER == 'PASS') & 
           (substr(SAMPLE,start=nchar(SAMPLE), stop=nchar(SAMPLE))) != 'B' &
           REF != ALT) %>% 
  mutate(MUTATION = paste(REF, ALT, sep = '>'))

npatients = length(unique(m.vnts$SAMPLE))

m.freq = m.vnts %>%
  group_by(POS, MUTATION) %>%
  summarise(FREQ = n()) %>% 
  mutate(AF = FREQ/npatients) %>%
  mutate(PGENIC = as.factor(
    ifelse(str_c(POS, MUTATION) %in% pgen$Allele, 1, 0))
    ) %>%
  mutate(POS = as.numeric(POS)) 

m.freq = rbind(m.freq, data.frame(POS = 152,
                                  MUTATION = 'T>TTC',
                                  FREQ = 1,
                                  AF = 0.1,
                                  PGENIC = as.factor(1)))
# dummy data to pretentd there is an insertion var at pos 152  which is pgenic 
# (in addition to the T>C SNP at 152)

m.freq = arrange(m.freq, POS)
nvnts = nrow(m.freq)

ukbb = read.table('ukbb_freqM.txt', header = TRUE) %>% filter(pop == 'EUR')
  # data from ukbb
bb.split = ukbb %>%
  separate(variant, into = c('chrom', 'POS', 'MUTATION'), sep = ':', 
           remove = FALSE) %>%
  mutate(MUTATION = gsub(',', '>', MUTATION))

ukbb = ukbb %>% rename(MUTATION = variant) %>%
  mutate(chrom = bb.split$chrom, 
         POS = as.numeric(bb.split$POS), 
         MUTATION = bb.split$MUTATION, 
         .before = MUTATION)
n.ukbb = round(mean(ukbb$N/ukbb$AF, na.rm = TRUE))

ukbb.freq = ukbb[,c('POS', 'MUTATION', 'N', 'AF')] %>%
  mutate(PGENIC = as.factor(
    ifelse(str_c(POS, MUTATION) %in% pgen$Allele, 1, 0))
    ) %>%
  mutate(POS = as.numeric(POS)) %>%
  arrange(POS)

ukbb.compare = ukbb.freq %>% 
  semi_join(m.freq, by = c('POS', 'MUTATION'))

ctrl = m.freq

sample.pgen = semi_join(m.freq, pgen, by = 'POS')
ukbb.pgen = semi_join(ukbb, pgen, by = 'POS')
ctrl.pgen = semi_join(ctrl, pgen, by = 'POS')
  # pathogenic variants in sample and ukbb data

m.gene = read.csv('mtgenes.csv')
  # data from source data of "Deleterious heteroplasmic mitochondrial mutations are associated with an increased risk of overall and cancer-specific mortality"

m.genelims = m.gene %>%
  group_by(GENE) %>%
  summarise(start = min(POS),
            end = max(POS)) %>%
  relocate(GENE, .after = 'end')
ngenes = nrow(m.genelims)

m.genelims = mutate(m.genelims, chr = rep('chrM', ngenes), .before = 'start')






# Double Checks
filter(m.freq, !POS %in% unique(ukbb.compare$POS))
  # checks for any variants in sample which are not in ukbb

m.vnts %>%
  group_by(POS) %>%
  summarise(FREQ = n())
  # nrow([above]) = nrow(m.freq) = 103, so there are no two variants with
  # the same position





# Residuals 
vnt.compare = data.frame(POS = m.freq$POS,
                         sampleAF = m.freq$AF,
                         ukbbAF = ukbb.compare$AF,
                         resid = m.freq$AF - ukbb.compare$AF,
                         resid.sq = (m.freq$AF - ukbb.compare$AF)^2,
                         PGENIC = m.freq$PGENIC)
m.RMSE = sqrt(sum(vnt.compare$resid.sq)/nrow(vnt.compare))
  # Root Mean square Error of the sample AF

alpha = 1.5 # adjusts limit for considering data as an outlier
m.outliers = vnt.compare %>%
  filter(resid >= alpha*m.RMSE)

m.outliers = m.outliers %>%  
  mutate(outlier = as.factor(rep(1, nrow(m.outliers))))
  # all variants in m. with an AF significantly different to the corresponding
  # variant in ukbb (significance determined by alpha)




# Scatter plot of UKBB AF against sample AF

# check for a difference in correlation between sampleAF and ukbbAF (effectively
# the actual AF) when pathogenic variants are removed
afplot = data.frame(POS = ukbb.compare$POS,
                    sampleAF = m.freq$AF,
                    ukbbAF = ukbb.compare$AF,
                    PGENIC = ukbb.compare$PGENIC) %>%
  mutate(outlier = POS %in% m.outliers$POS)

pgenic.pos = c(sample(afplot$POS, 10))
  # add positions of variants to remove to compare corrs in the plot
  # currently just takes 10 random variants
afplot.nopgen = afplot %>%
  filter(!POS %in% pgenic.pos)

lmodel = lm(sampleAF ~ ukbbAF, afplot)
lmodel.summ = summary(lmodel)

noP_lmodel = lm(sampleAF ~ ukbbAF, afplot.nopgen)
noP_lmodel.summ = summary(noP_lmodel)

cor(afplot$sampleAF, afplot$ukbbAF)

ggplot(afplot, aes(ukbbAF, sampleAF, col = PGENIC)) +
  geom_point(pch = 1) +
  geom_point(data = m.outliers, col = 'black', pch = 4) +
  geom_abline(slope = lmodel$coefficients[2], 
              intercept = lmodel$coefficients[1]) +
  geom_abline(slope = noP_lmodel$coefficients[2],
              intercept = noP_lmodel$coefficients[1],
              col = 'red')
  # currently not too reliable for low AF variants in small sample data sizes

  # [ukbb data has AF << 0.1 but in samples with say only n=10, the AF is either
  # 0 or 0.1 so could be counted as an outlier when it is just a regular
  # harmless variant to be expected in each person]

alpha = 1.5
in.alpha.sd = c(lmodel$coefficients[2] - alpha*lmodel.summ$coefficients[2,2] <=
                  noP_lmodel$coefficients[2] &
                  noP_lmodel$coefficients[2] <=
                  lmodel$coefficients[2] + alpha*lmodel.summ$coefficients[2,2])
  # returns TRUE or FALSE depending on if the noP model has a slope who's value
  # is within alpha standard deviations of the slope of the full data model

all.cor = lmodel.summ$r.squared
noP.cor = noP_lmodel.summ$r.squared
corr.diff = lmodel.summ$r.squared - noP_lmodel.summ$r.squared
  # difference in linear correlation between whole data and noP data

# could make a loop to compare removal of all possible combinations of pgenic 
# variants







# Circos Plots
circos.initialize('chrM', xlim = c(0, 16569), ring = TRUE) 
  # the 'chrM' acts as the label for the single sector in circular genome plot

circos.genomicLabels(m.genelims, labels.column = 4, 
                     labels.side = 'outside', 
                     cex = 0.6,
                     labels_height = mm_h(5),
                     connection_height = mm_h(0.1),
                     line_lty = 'blank',
                     padding = 0,
                     track.margin = c(0.16, 0.01)) 
circos.track(ylim = c(0, 1), 
             track.height = 0.08,
             bg.col = 'grey93',
             track.margin = c(0.01, 0),
             cell.padding = c(0,0,0,0),
             panel.fun = function(x, y) {
               palette('default')
               circos.xaxis(direction = 'outside', labels.cex = 0.6)
               circos.rect(xleft = m.genelims$start,
                           xright = m.genelims$end,
                           ybottom = rep(0, ngenes),
                           ytop = rep(1, ngenes),
                           col = as.factor(m.genelims$GENE),
               )
             })
circos.track(ylim = c(0,1),
             bg.col = 'grey93',
             track.margin = c(0.01, 0),
             panel.fun = function(x,y) {
               palette(c('darkseagreen', 'red4'))
               circos.barplot(value = ukbb.freq$AF,
                              pos = as.numeric(ukbb.freq$POS),
                              col = ukbb.freq$PGENIC, 
                              border = ukbb.freq$PGENIC,)
               circos.points(x = as.numeric(ukbb.pgen$POS),
                             y = rep(0.85, nrow(ukbb.pgen)),
                             pch = 5,
                             col = 'red4')
             })
circos.track(ylim = c(0,1),
             bg.col = 'grey93',
             track.height = 0.15,
               panel.fun = function(x,y){
                 
                 palette(c('slateblue', 'red4'))
                 circos.barplot(value = m.freq$AF, 
                                pos = as.numeric(m.freq$POS),
                                col = m.freq$PGENIC, 
                                border = m.freq$PGENIC)
                 circos.points(x = as.numeric(sample.pgen$POS),
                               y = rep(0.85, nrow(sample.pgen)),
                               pch = 5,
                               col = 'red4')
               })
circos.track(ylim = c(0,1),
             bg.col = 'grey93',
             track.height = 0.15,
             panel.fun = function(x,y){
               
               palette(c('lightseagreen', 'red4'))
               circos.barplot(value = m.freq$AF, 
                              pos = as.numeric(ctrl$POS),
                              col = ctrl$PGENIC, 
                              border = ctrl$PGENIC)
               circos.points(x = as.numeric(sample.pgen$POS),
                             y = rep(0.85, nrow(sample.pgen)),
                             pch = 5,
                             col = 'red4')
             })

palette('default')
legend('bottomleft', 
       legend = c('(present) pathogenic variants'), 
       cex = 0.8, pch = 5, col = 'red4')
legend('bottomright', 
       legend = c('ukbb', 'sample', 'control'),
       cex = 0.8, fill = c('darkseagreen', 'slateblue', 'lightseagreen'))

title('Frequencies of Variants')
circos.clear()

