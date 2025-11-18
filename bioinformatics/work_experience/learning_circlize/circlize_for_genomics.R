library(circlize)

# Human
circos.initializeWithIdeogram(species = 'hg19') # default df is cytoband
                            # data of human genome 'hg19'
text(0, 0, "Human Cytoband", cex = 1)
circos.info() # shows info about current circos plot:
                        # sectors, tracks, and selected cell
circos.clear()


# Subset of Human
circos.initializeWithIdeogram(chromosome.index = paste0("chr", c(3,5,2,8)))
text(0, 0, "subset of chromosomes", cex = 1)
circos.clear()
  # only displays chromosomes 2, 3, 5, and 8


# Mouse
mouse.cyto = read.cytoband(species = 'mm10')
mouse.cyto.df = mouse.cyto$df
  # how to get cytoband and df from UCSC (in this case for mice)

circos.initializeWithIdeogram(mouse.cyto.df)
text(0,0, 'Mouse from imported df')
circos.clear()

circos.initializeWithIdeogram(species = 'mm10')
text(0,0, 'Mouse straight from UCSC')
circos.clear()
  # these two plots are the same so safe to use either 
  # [imported preferable for exploration]


