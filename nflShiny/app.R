# Install and load required packages
# install.packages("shiny")
# install.packages("tidyverse")
# install.packages("gt")
# install.packages("nflverse")
# install.packages("dplyr")

library(shiny)
library(tidyverse)
library(gt)
library(nflverse)
library(dplyr)

ui <- fluidPage(
    titlePanel("NFL Analysis App"),
    
    mainPanel(
        navlistPanel(
            widths = c(2, 10),
            tabPanel("Steelers Point Differential",
                     h3("Pittsburgh Steelers Total Point Differential"),
                     textOutput("total_point_differential")
            ),
            tabPanel("Team Performance Summary",
                     h3("NFL Team Performance Summary"),
                     gt_output("team_summary_table")
            ),
            tabPanel("Elo-Style Rankings",
                     h3("NFL Team Rankings Considering Opponent Strength"),
                     gt_output("final_rankings_table")
            ),
            tabPanel("Next Game Prediction",
                     h3("Next Pittsburgh Steelers Game Prediction"),
                     textOutput("next_game_info"),
                     textOutput("win_probability"),
                     textOutput("monte_carlo_probability")
            )
        )
    )
)

server <- function(input, output) {
    
    # Load game schedules for the current season
    game_outcomes <- reactive({
        load_schedules(season = 2024)
    })
    
    # Filter Pittsburgh Steelers games
    steelers_games <- reactive({
        game_outcomes() %>% filter(home_team == "PIT" | away_team == "PIT")
    })
    
    # Calculate Steelers' total point differential
    steelers_games_filtered <- reactive({
        steelers_games() %>%
            filter(!is.na(home_score) & !is.na(away_score)) %>%
            mutate(
                points_scored = ifelse(home_team == "PIT", home_score, away_score),
                points_allowed = ifelse(home_team == "PIT", away_score, home_score),
                point_differential = points_scored - points_allowed
            )
    })
    
    total_point_differential <- reactive({
        sum(steelers_games_filtered()$point_differential)
    })
    
    output$total_point_differential <- renderText({
        total_point_differential()
    })
    
    # Team performance summary
    completed_games <- reactive({
        game_outcomes() %>%
            filter(!is.na(home_score) & !is.na(away_score))
    })
    
    team_summary <- reactive({
        home_stats <- completed_games() %>%
            mutate(
                team = home_team,
                points_scored = home_score,
                points_allowed = away_score,
                win = ifelse(home_score > away_score, 1, 0),
                loss = ifelse(home_score < away_score, 1, 0)
            ) %>%
            select(team, points_scored, points_allowed, win, loss)
        
        away_stats <- completed_games() %>%
            mutate(
                team = away_team,
                points_scored = away_score,
                points_allowed = home_score,
                win = ifelse(away_score > home_score, 1, 0),
                loss = ifelse(away_score < home_score, 1, 0)
            ) %>%
            select(team, points_scored, points_allowed, win, loss)
        
        team_stats <- bind_rows(home_stats, away_stats)
        
        team_stats %>%
            group_by(team) %>%
            summarise(
                point_differential = sum(points_scored) - sum(points_allowed),
                wins = sum(win),
                losses = sum(loss)
            ) %>%
            ungroup() %>%
            arrange(desc(point_differential))
    })
    
    output$team_summary_table <- render_gt({
        team_summary() %>%
            gt() %>%
            tab_header(
                title = "NFL Team Performance Summary",
                subtitle = "Point Differential, Wins, and Losses for the Current Season"
            ) %>%
            cols_label(
                team = "Team",
                point_differential = "Point Differential",
                wins = "Wins",
                losses = "Losses"
            ) %>%
            fmt_number(
                columns = c(point_differential, wins, losses),
                decimals = 0
            ) %>%
            tab_style(
                style = cell_text(weight = "bold"),
                locations = cells_column_labels(everything())
            ) %>%
            data_color(
                columns = point_differential,
                colors = scales::col_numeric(
                    palette = c("red", "white", "green"),
                    domain = c(min(team_summary()$point_differential), max(team_summary()$point_differential))
                )
            ) %>%
            tab_options(
                table.font.size = "medium",
                table.width = pct(100)
            )
    })
    
    # Elo-style team rankings using your code
    final_rankings <- reactive({
        # Ensure 'week' is numeric and arrange games by week
        completed_games <- game_outcomes() %>%
            filter(!is.na(home_score) & !is.na(away_score)) %>%
            mutate(week = as.numeric(week)) %>%
            arrange(week)
        
        # Initialize ratings for all teams (starting rating 1500 for all)
        initial_rating <- 1500
        ratings <- team_summary() %>%
            mutate(rating = initial_rating)
        
        # Get a list of unique weeks to iterate over
        unique_weeks <- unique(completed_games$week)
        
        # Iterate over each week to update ratings
        for (current_week in unique_weeks) {
            # Filter games for the current week
            weekly_games <- completed_games %>%
                filter(week == current_week)
            
            # Join the current ratings to the weekly games to calculate adjustments
            weekly_games_with_ratings <- weekly_games %>%
                left_join(ratings, by = c("home_team" = "team")) %>%
                rename(home_rating = rating) %>%
                left_join(ratings, by = c("away_team" = "team")) %>%
                rename(away_rating = rating)
            
            # Calculate expected outcomes and rating adjustments
            weekly_games_with_ratings <- weekly_games_with_ratings %>%
                mutate(
                    home_expected = 1 / (1 + 10 ^ ((away_rating - home_rating) / 400)),
                    away_expected = 1 - home_expected,
                    home_result = ifelse(home_score > away_score, 1, 0),
                    away_result = ifelse(away_score > home_score, 1, 0),
                    home_rating_change = 20 * (home_result - home_expected),  # K-factor of 20 for update
                    away_rating_change = 20 * (away_result - away_expected)
                )
            
            # Update ratings based on weekly results
            ratings <- ratings %>%
                left_join(
                    weekly_games_with_ratings %>%
                        group_by(home_team) %>%
                        summarise(rating_change = sum(home_rating_change)) %>%
                        rename(team = home_team),
                    by = "team"
                ) %>%
                mutate(rating = rating + coalesce(rating_change, 0)) %>%
                select(-rating_change) %>%
                left_join(
                    weekly_games_with_ratings %>%
                        group_by(away_team) %>%
                        summarise(rating_change = sum(away_rating_change)) %>%
                        rename(team = away_team),
                    by = "team"
                ) %>%
                mutate(rating = rating + coalesce(rating_change, 0)) %>%
                select(-rating_change)
        }
        
        # Create a final dataframe of rankings with rank column
        final_rankings <- ratings %>%
            arrange(desc(rating)) %>%
            mutate(rank = row_number())
        
        final_rankings
    })
    
    output$final_rankings_table <- render_gt({
        final_rankings() %>%
            gt() %>%
            tab_header(
                title = "NFL Team Rankings Considering Opponent Strength",
                subtitle = "Elo-Style Ratings for Current Season, Updated Week-by-Week"
            ) %>%
            cols_label(
                team = "Team",
                rating = "Rating",
                rank = "Rank"
            ) %>%
            fmt_number(
                columns = rating,
                decimals = 1
            ) %>%
            tab_style(
                style = cell_text(weight = "bold"),
                locations = cells_column_labels(everything())
            )
    })
    
    # Next Steelers game prediction (remains the same)
    next_game_info <- reactive({
        # Get the last completed game date
        completed_games <- completed_games()
        completed_games <- completed_games %>%
            mutate(gameday = as.Date(gameday))
        last_game_date <- max(completed_games$gameday, na.rm = TRUE)
        
        # Find the next Steelers game
        next_game <- game_outcomes() %>%
            mutate(gameday = as.Date(gameday)) %>%
            filter((home_team == "PIT" | away_team == "PIT") & gameday > last_game_date) %>%
            arrange(gameday) %>%
            slice(1)
        
        if (nrow(next_game) == 0) {
            list(
                game_info = "No game scheduled for the next week for Pittsburgh Steelers.",
                win_prob = NULL,
                monte_carlo_prob = NULL
            )
        } else {
            if (next_game$home_team == "PIT") {
                opponent_team <- next_game$away_team
            } else {
                opponent_team <- next_game$home_team
            }
            
            # Get Steelers and opponent ratings
            ratings <- final_rankings()
            steelers_rating <- ratings %>%
                filter(team == "PIT") %>%
                pull(rating)
            opponent_rating <- ratings %>%
                filter(team == opponent_team) %>%
                pull(rating)
            
            # Calculate win probability
            steelers_win_probability <- 1 / (1 + 10 ^ ((opponent_rating - steelers_rating) / 400))
            
            # Monte Carlo simulation
            set.seed(42)
            num_simulations <- 10000
            simulated_outcomes <- rbinom(num_simulations, 1, steelers_win_probability)
            simulated_win_probability <- mean(simulated_outcomes)
            
            list(
                game_info = paste("Pittsburgh Steelers next game is against:", opponent_team,
                                  "on", as.character(next_game$gameday)),
                win_prob = paste("Pittsburgh Steelers win probability:", round(steelers_win_probability * 100, 2), "%"),
                monte_carlo_prob = paste("Monte Carlo Simulated Win Probability for Pittsburgh Steelers:", round(simulated_win_probability * 100, 2), "%")
            )
        }
    })
    
    output$next_game_info <- renderText({
        next_game_info()$game_info
    })
    
    output$win_probability <- renderText({
        next_game_info()$win_prob
    })
    
    output$monte_carlo_probability <- renderText({
        next_game_info()$monte_carlo_prob
    })
}

shinyApp(ui = ui, server = server)
