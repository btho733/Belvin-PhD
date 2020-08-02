# This is an app that shows the data, methods and results of a color based clustering strategy
# developed to address the class imbalance problem.
#
# To go to the web-app : Click on the link given below
# https://belsys.shinyapps.io/AppDir/
#
# Short description of project :
# A novel approach has been developed to implement the popular k-means clustering in the presence of extreme
# class imbalance. After clustering all the slices in a 3D stack of images into 3 groups (fibrosis, myocytes and fat), 
# the slice-wise distribution of these tissue types is analysed. Significant regions are located from the bumps in the 
# plotly graph. These sites are then used for furtheranalysis. Additionally, the proportion of tissue types in the 
# selected slice is compared with the regional and overall proportions. See the Implememtation details tab in the app 
# for more info.
#
# For a detailed documentation of the project:
# Email me for a copy of thesis chapter btho733@aucklanduni.ac.nz
#
# Author : Belvin Thomas

library(shiny)
library(R.matlab)
library(RcppRoll)
library(ggplot2)
library(plotly)

ui <- fluidPage(# navbarPage(title= span(h3("Color based clustering of imbalanced classes (Quantification of atrial fibrosis)"),
  # style = "background-color: #DEEBF7;color: black"),windowTitle="Color based clustering for imbalanced classes",
  wellPanel(
    h1(
      "Color based clustering of imbalanced classes (Quantification of atrial tissue types)"
    )
  ),
  tabsetPanel(
    tabPanel(
      "Cluster Analysis",
      fluidRow(#column(2,selectInput(inputId = "n",
        #                    label="Select Slice no.",
        #                   choices= 83:1011,
        #                  selected=580))
        
        column(
          2,
          sliderInput(
            inputId = "n",
            label = "Select Slice no.",
            min = 83,
            max = 1011,
            value = 580
          )
        )),
      fluidRow(
        column(3, plotOutput("plot0", width = "100%")),
        column(3, plotOutput("plot1", width = "100%")),
        column(3, plotOutput("plot3", width = "100%")),
        column(3, plotOutput("plot2", width = "auto"))
      ),
      fluidRow(
        column(5, plotlyOutput("plot4", width = "auto")),
        column(4, plotlyOutput("plot5", width = "auto")),
        column(3, tags$img(
          height = 400,
          width = 350,
          src = "Legend.png"
        ))
      )
    ),
    tabPanel(
      "Implementation details",
      tags$h4(tags$strong("Background")),
      fluidRow(column(
        12,
        p(
          style = "text-align:justify",
          "Identification and quantification of connective tissue over the atrial volume are topics of
considerable interest among researchers studying the relationship between atrial structure and electrical function (Nattel and Harada, 2014). Numerous experimental and
clinical studies, have demonstrated a significant association between fibrosis and atrial
rhythm disturbance, especially when the abnormality is accompanied by heart failure or
atrial dilatation (Boixel et al., 2003; Hugh et al., 2015). Tissue specific modelling has
been used to understand the role played by fibrosis in the initiation and maintenance
of atrial electrical dysfunction (McDowell et al., 2013; Zhao et al., 2017). Accurate
identification of tissue types is highly significant in the design of such modelling studies).
            "
        )
      )),
      
      fluidRow(column(
        12,
        p(
          style = "text-align:justify",
          "This page summarises the development of a novel color-based clustering app (see the cluster analysis tab)
  to segment the types of tissue observed in a dilated sheep atria - fibrosis, myocytes and fat. The high resolution dataset is made up
  of a 3D stack (more than 1000 slices) of stained images. After ventricular tachypacing to induce heart disease, an  entire sheep atria have been digitised
  from top to bottom by using an imaging modality called",
          tags$u(
            "serial block face imaging technique (A representative slice from the stack is shown below)"
          ),
          ". The three types of tissue are specifically identified
  by topical application of May-Grunwald stain",
          tags$strong(
            "[Fibrosis (Collagen) is coloured purplish, Myocytes are coloured greenish and Fat cells are coloured yellowish.]"
          )
        )
      )),
      tags$br(),
      fluidRow(column(
        6, tags$img(
          height = 500,
          width = 720,
          src = "fig5_1.png"
        )
      ),
      column(
        4, offset = 1, tags$img(
          height = 500,
          width = 400,
          src = "fig5_4.png"
        )
      )),
      tags$br(),
      tags$h4(tags$strong("The problem")),
      fluidRow(
        column(
          7,
          p(
            style = "text-align:justify",
            "In this data set, myocytes and fat cells make up the lion's share of the structure, although their distribution may vary
unpredictably between slices. Consequently, the purple-stained fibrotic tissue represents a small proportion of the total tissue volume. This is a common trend in most of the
scientific/medical datasets where the most significant class turns out to be the minority class. Efficient ways of grouping this type of data are actively studied in the clustering
literature under the tag of ",
            tags$strong("class imbalance problems"),
            ". Severely skewed distributions are notoriously known for damaging the efficacy of machine learning based workflows,
particularly when the available data holds the key (Japkowicz et al., 2002; Krawczyk, 2016; Weiss et al., 2001). In this context, the stark imbalance between fibrotic tissue and
the other two tissue types needs to be treated with due seriousness. A close observation of the images also reveals that fibrotic tissue does not form concentrated patches, but is scattered across
the atria as very fine lines or diffuse spots. In addition to these issues associated with extreme class imbalance, the colour information carried by each of the tissue types varies due to differences in staining
and lighting conditions. Although resultant artefact is reduced by pre-processing steps including image normalisation, the three tissue types are characterised by various shades of purple, green and yellow.",
            tags$u(
              "The figure showing ",
              tags$strong("color distribution in 3 color spaces (RGB, HSB and CIELAB)")
            ),
            " presents the variety of shades present in a representative slice taken from the stack.
                                      "
          ),
          tags$br(),
          tags$h4(tags$strong("The solution")),
          fluidRow(column(
            12,
            p(
              style = "text-align:justify",
              "K-means clustering in CIELAB color space is a popular approach for color-based segmentation.
  However, it is well known that the performance of k-means algorithm is largely dependent on initialisation. Due to the algorithm's unsupervised nature, it could fall into a local optimum depending on the initial cluster centroids.
  Usually the process is randomly repeated multiple times to overcome this problem. Though, it is very much possible that all the random trials could miss the global optimum, returning unsatisfactory clustering results every time.
  It was observed that the presence of class imbalance further degrades the performance and messes up the segmenatation
  results delivered by this approach. ",
              tags$u(
                "Refer to the figures on the right side panel to see the ",
                tags$strong(
                  "clustering results(k=3) produced by 3 different initialisations"
                )
              ),
              ". The third set shows better segmentations compared to the other two sets.
  Hence, the question addressed here is whether we can find the right set of initial centroids for every slice from top to bottom of the stack. This is necessarily intended to ensure robust segmentation of tissue types in the whole volume.
  The solution is powered by a resampling process which produces a balanced distribution of the three dominant colors present in each slice. ",
              tags$u(
                "The approach developed to ",
                tags$strong("filter out a*-b* plot"),
                " is summarised in the figure below"
              ),
              ". From this two dimensional representation of color gamut, the tissue sectors are estimated based on their respective hue values. Then, the centre of mass
  of these estimated sectors are used to initialise k-means clustering in each slice. "
            )
          )),
          tags$br(),
        ),
        column(
          5,
          tags$br(),
          fluidRow(column(
            10, tags$img(
              height = 150,
              width = 600,
              src = "fig5_2.png"
            )
          )),
          fluidRow(column(
            10, tags$img(
              height = 150,
              width = 600,
              src = "fig5_3.png"
            )
          )),
          fluidRow(column(
            10, tags$img(
              height = 150,
              width = 600,
              src = "fig5_7.png"
            )
          )),
          
        )
      ),
      tags$br(),
      
      
      fluidRow(column(
        11,
        offset = 1,
        tags$img(
          height = 300,
          width = 1300,
          src = "fig5_5.png"
        )
      )),
      tags$br(),
      tags$h4(tags$strong("Results")),
      fluidRow(column(
        12,
        p(
          style = "text-align:left",
          "The interactive dashboard provides a set of insights which further leads to major results/observations as outlined below:  "
        )
      )),
      fluidRow(column(
        12,
        p(
          style = "text-align:left",
          tags$strong("1. Robust segmentation of tissue types from top to bottom of stack"),
          " : This fully automatic approach enabled extraction of 3D collagen network(Fibrosis) from the stack (See figure below)."
        )
      )),
      fluidRow(column(
        12,
        p(
          style = "text-align:left",
          tags$strong("2. Accurate quantification of the three types of tissue"),
          " :  The visualisations provided a graphical comparison of tissue proportions at various levels - slice level, regional and overall. "
        )
      )),
      fluidRow(column(
        12,
        p(
          style = "text-align:left",
          tags$strong("3. Objective local analysis"),
          " : The bumps (red rectangles) in the graph helped to locate the most fibrotic regions from Pulmonary Vein sleeves(figure A below) and lateral wall of Right Atrium(figure B below). This motivated further local analysis (See figures C and D below).   "
        )
      )),
      tags$br(),
      fluidRow(
        column(
          5,
          offset = 1,
          tags$img(
            height = 400,
            width = 650,
            src = "fig5_8_1.jpg"
          )
        ),
        column(5, offset = 1, fluidRow(column(
          10, tags$img(
            height = 200,
            width = 400,
            src = "Annotation.png"
          )
        )),
        fluidRow(column(
          10, tags$img(
            height = 200,
            width = 400,
            src = "fig5_9.png"
          )
        )))
      ),
      tags$br(),
    )
    
  ))

server <- function(input, output, session) {
  # Original image (selected slice)
  output$plot0 <- renderImage({
    original <- normalizePath(file.path(
      './images/original',
      paste('s10_', sprintf("%05d", strtoi(input$n)), '.png', sep =
              '')
    ))
    
    
    width  <- session$clientData$output_plot0_width
    height <- 0.7 * session$clientData$output_plot0_width
    # Return a list containing the filename
    list(src = original,
         width = width,
         height = height)
  }, deleteFile = FALSE)
  
  # Segmented geometry (selected slice)
  output$plot1 <- renderImage({
    filename <- normalizePath(file.path('./images/geo',
                                        paste(
                                          'seg_', sprintf("%05d", strtoi(input$n)), '.png', sep = ''
                                        )))
    
    width  <- session$clientData$output_plot1_width
    height <- 0.7 * session$clientData$output_plot1_width
    # Return a list containing the filename
    list(src = filename,
         width = width,
         height = height)
  }, deleteFile = FALSE)
  # The filtered a*-b* plot showing a balnced color distribution of currently selected slice
  output$plot2 <- renderPlot({
    uabfile <- normalizePath(file.path('./images/uab',
                                       paste(
                                         'uab_', sprintf("%05d", strtoi(input$n)), '.mat', sep = ''
                                       )))
    # For reading mat files
    uab <- readMat(uabfile)
    a <- uab$uab[, 1]
    b <- uab$uab[, 2]
    r <- uab$uab[, 3]
    g <- uab$uab[, 4]
    blue <- uab$uab[, 5]
    h <- rgb(r, g, blue, maxColorValue = 255)
    plot(
      a,
      b,
      pch = 15,
      col = h,
      xlab = "a*",
      ylab = "b*",
      font.lab = 2,
      cex.lab = 1.2
    )
    title('Balanced color distribution')
  },
  height = function()
    0.75 * session$clientData$output_plot2_height)
  # Segmented tissue types after clustering into 3 groups -fibrosis (purple), myocytes (green) and fat (yellow)
  output$plot3 <- renderImage({
    segname <- normalizePath(file.path('./images/seg',
                                       paste(
                                         'Im_', sprintf("%05d", strtoi(input$n)), '.png', sep = ''
                                       )))
    
    width  <- session$clientData$output_plot3_width
    height <- 0.7 * session$clientData$output_plot3_width
    # Return a list containing the filename
    list(src = segname,
         width = width,
         height = height)
  }, deleteFile = FALSE)
  
  # Plotly graph (slice-wise display of the quantity of each tissue type (percentages))
  output$plot4 <- renderPlotly({
    percentcsv <-
      round(read.csv("./images/percent_CollTissFat_merged.csv", header = F),
            2)
    mavgfib = roll_mean(percentcsv$V2, 40, fill = 0)
    mavgtiss = roll_mean(percentcsv$V3, 40, fill = 0)
    mavgfat = roll_mean(percentcsv$V4, 40, fill = 0)
    data1 <- percentcsv[83:1011, ]
    Local_Fibrosis <- mavgfib[83:1011]
    Local_Myocytes <- mavgtiss[83:1011]
    Local_Fat <- mavgfat[83:1011]
    Slice <- data1$V1
    Fibrosis <- data1$V2
    Myocytes <- data1$V3
    Fat <- data1$V4
    p1 <- ggplot(data = data1) +
      geom_line(colour = colors()[50], aes(Slice, Myocytes)) +
      geom_line(colour = colors()[84], aes(Slice, Fibrosis)) +
      geom_line(colour = colors()[24], aes(Slice, Local_Fibrosis)) +
      geom_line(colour = colors()[24], aes(Slice, Local_Myocytes)) +
      geom_line(colour = colors()[24], aes(Slice, Local_Fat)) +
      geom_line(colour = colors()[142], aes(Slice, Fat)) +
      labs(title = "Slice-wise quantification of atrial tissue types", x = "Slice Number", y = "Percentage") +
      theme(
        plot.title = element_text(
          color = "black",
          size = 10,
          face = "bold",
          hjust = 0.5
        ),
        axis.title.x = element_text(
          color = "black",
          size = 10,
          face = "bold"
        ),
        axis.text.x = element_text(
          color = "black",
          size = 10,
          face = "plain"
        ),
        axis.text.y = element_text(
          color = "black",
          size = 10,
          face = "plain"
        ),
        axis.title.y = element_text(
          color = "black",
          size = 10,
          face = "bold"
        )
      )
    
    ggplotly(p1)
  },)
  # Interactive Bar chart (comparing the tissue proportionsat 3 levels- slice, regional and overall)
  output$plot5 <- renderPlotly({
    percentcsv <-
      round(read.csv("./images/percent_CollTissFat_merged.csv", header = F),
            2)
    mavgfib = roll_mean(percentcsv$V2, 40, fill = 0)
    mavgtiss = roll_mean(percentcsv$V3, 40, fill = 0)
    mavgfat = roll_mean(percentcsv$V4, 40, fill = 0)
    Location <-
      c(rep("Selected slice", 3),
        rep("Regional", 3),
        rep("Overall", 3))
    Tissue_Type <- rep(c("Fibrosis", "Myocytes", "Fat"), 3)
    Percentage <-
      c(
        percentcsv$V2[strtoi(input$n)],
        percentcsv$V3[strtoi(input$n)],
        percentcsv$V4[strtoi(input$n)],
        mavgfib[strtoi(input$n)],
        mavgtiss[strtoi(input$n)],
        mavgfat[strtoi(input$n)],
        mean(percentcsv$V2[83:1011]),
        mean(percentcsv$V3[83:1011]),
        mean(percentcsv$V4[83:1011])
      )
    data <- data.frame(Location, Tissue_Type, Percentage)
    
    data$Location <-
      factor(data$Location,
             levels = c('Selected slice', 'Regional', 'Overall'))
    data$Tissue_Type <-
      factor(data$Tissue_Type, levels = c('Fibrosis', 'Myocytes', 'Fat'))
    
    p2 <-
      ggplot(data, aes(fill = Tissue_Type, y = Percentage, x = Location)) +
      geom_bar(position = "dodge", stat = "identity") +
      scale_fill_manual(values = colors()[rep(c(84, 50, 142), 3)]) +
      labs(title = "Location-wise comparison", x = "Locations", y = "Percentage") +
      theme(
        plot.title = element_text(
          color = "black",
          size = 10,
          face = "bold",
          hjust = 0.5
        ),
        axis.title.x = element_text(
          color = "black",
          size = 10,
          face = "bold"
        ),
        axis.text.x = element_text(
          color = "black",
          size = 9,
          face = "plain"
        ),
        axis.text.y = element_text(
          color = "black",
          size = 10,
          face = "plain"
        ),
        axis.title.y = element_text(
          color = "black",
          size = 10,
          face = "bold"
        )
      )
    ggplotly(p2)
  })
  
}


shinyApp(ui = ui, server = server)
