library(circlize)

set.seed(999)
n = 1000
df = data.frame(sectors = sample(letters[1:8], n, replace = TRUE),
                x = rnorm(n),
                y = runif(n))

circos.par('track.height' = 0.1)
  # parameters for plot
circos.initialize(df$sectors, x = df$x)
  # initializes the data
circos.track(df$sectors, y = df$y,
  # circos.track describes what should be in the tracks in the plot
             panel.fun = function(x, y){  # adjusts display of plot
                                          # does this one cell at a time
               circos.text(x = CELL_META$xcenter, 
  # CELL_META gives info about current cell [check names(CELL_META)]
              # x = describes x pos of text                   
                          y = CELL_META$cell.ylim[2] + mm_y(5),
              # y = y pos of text + mm_y(5) adds 5mm of space in y dir to text
              # CELL_META$cell.ylim[2] = outer lim of radius, [1] = inner lim
                           labels = CELL_META$sector.index)
              # labels = 'the text we want to display'
              # C_M$sector.index uses the sector names as the labels
               circos.axis(labels.cex = 0.6)
                # .cex = size of related text
             })
col = rep(c("#FF0000", "#00FF00"), 4)
circos.trackPoints(df$sectors, df$x, df$y, col = col, pch = 16, cex = 0.5)
circos.text(-1, 0.5, "text", sector.index = "a", track.index = 1)

bgcol = rep(c("#EFEFEF", "#CCCCCC"), 4)
  # background colours for the histogram
circos.trackHist(df$sectors, x = df$x, bin.size = 0.2, bg.col = bgcol)
  # bin.size makes the width (bins) smaller/larger

circos.track(df$sectors, x = df$x, y = df$y,
             panel.fun = function(x, y) {
               ind = sample(length(x), 10)
               x2 = x[ind]
               y2 = y[ind]
               od = order(x2)
               circos.lines(x2[od], y2[od])
      # this panel.fun uses the idea of 'one cell at a time' to match
      # 10 random x and y values in each cell and uses them for a line plot
             })

circos.update(sector.index = "d", track.index = 2, 
              bg.col = "#FF8080", bg.border = "black")
circos.points(x = -2:2, y = rep(0.5, 5), col = "white")
circos.text(CELL_META$xcenter, CELL_META$ycenter, "updated", col = "white")
  # changes only the track 2 sector d cell into a points plot of just 5 points
  # (x, 0.5) for x =-2,-1,0,1,2 with text 'updated' in the center of the cell
  # [specified by CELL_META$xcenter, CELL_META$ycenter]

  # this also makes the currently selected cell the track 2 sector d cell

circos.track(ylim = c(0, 1), panel.fun = function(x, y) {
  xlim = CELL_META$xlim
  ylim = CELL_META$ylim
  breaks = seq(xlim[1], xlim[2], by = 0.1)
  n_breaks = length(breaks)
  circos.rect(breaks[-n_breaks], rep(ylim[1], n_breaks - 1),
              breaks[-1], rep(ylim[2], n_breaks - 1),
              col = rand_color(n_breaks), border = NA)
})
  # adds a heatmap [circos.rect()] with random colours, not related to df

circos.link("a", 0, "b", 0, h = 0.4)
circos.link("c", c(-0.5, 0.5), "d", c(-0.5,0.5), col = "red",
            border = "blue", h = 0.2)
circos.link("e", 0, "g", c(-1,1), col = "green", border = "black", 
            lwd = 2, lty = 2)
  # adds links between 1) 2 points, 2) 2 intervals, 3) a point and an interval

circos.clear()
  # resets all parameters and internal variables so next plot isn't messed up
  # 'cleans up'



  # barplots on different tracks next to a stacked barplot
par(mfrow = c(1, 2))
circos.initialize(letters[1:4], xlim = c(0, 10))
circos.track(ylim = c(0, 1), panel.fun = function(x, y) {
  value = runif(10)
  circos.barplot(value, 1:10 - 0.5, col = 1:10)
})
circos.track(ylim = c(-1, 1), panel.fun = function(x, y) {
  value = runif(10, min = -1, max = 1)
  circos.barplot(value, 1:10 - 0.5, col = ifelse(value > 0, 2, 3))
})
circos.clear()

circos.initialize(letters[1:4], xlim = c(0, 10))
circos.track(ylim = c(0, 4), panel.fun = function(x, y) {
  value = matrix(runif(10*4), ncol = 4)
  circos.barplot(value, 1:10 - 0.5, col = 2:5)
})
circos.clear()




  # boxplots, one made with a for loop, 
  # one made by applying circos.boxplot() to a matrix of values
par(mfrow = c(1,2))
circos.initialize(letters[1:4], xlim = c(0, 10))
circos.track(ylim = c(0, 1), panel.fun = function(x, y) {
  for(pos in seq(0.5, 9.5, by = 1)) {
    value = runif(10)
    circos.boxplot(value = value, pos = pos)
  }
})
circos.clear()

circos.initialize(letters[1:4], xlim = c(0, 10))
circos.track(ylim = c(0, 1), panel.fun = function(x, y) {
  value = replicate(runif(10), n = 10, simplify = FALSE)
  circos.boxplot(value, 1:10 - 0.5, col = 1:10)
})
circos.clear()
  # can do same for circos.violin() for a violin plot






par(mfrow = c(1, 2))
circos.initialize(sectors = letters[1:8], xlim = c(0, 1))
circos.track(ylim = c(0, 1))
circos.labels(c("a", "a", "b", "b"), x = c(0.1, 0.12, 0.4, 0.6), 
              labels = c(0.1, 0.12, 0.4, 0.6))
circos.clear()

circos.initialize(sectors = letters[1:8], xlim = c(0, 1))
circos.labels(c("a", "a", "b", "b"), x = c(0.1, 0.12, 0.4, 0.6), 
              labels = c(0.1, 0.12, 0.4, 0.6),
              side = "outside")
circos.track(ylim = c(0, 1))
circos.clear()

  # circos.track creates two new tracks, one for the lines and one for the 
  # text/labels



