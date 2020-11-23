library(shiny)

ui <- fluidPage(
  
  h1("Data Visualization Lab work - 2"),
  h4("Box Plots visulaization for Fuel Consumption in city roads between 10-30 km"),
  h3("Work done by Ravinthiran Partheepan"),
  
  selectInput("p","Select Data",choices = setdiff(names(mtcars),"mpg")),
  plotOutput("myplot"))

server <- function(input, output, session) {
  
  output$myplot <- renderPlot({
    m <- paste0('mpg','~',input$p)
    boxplot(as.formula(m) , xlab = 'Fuel Consumption compared with Y label - km/gear/Co2 emission', ylab = 'Fuel Consumption in city roads in terms of km', data=mtcars)
  })
}

shinyApp(ui, server)