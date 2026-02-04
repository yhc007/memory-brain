//! Interactive TUI Dashboard for memory-brain
//! 
//! Navigate memories, view stats, and search interactively! 🧠

use std::io::{self, stdout};
use std::time::Duration;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::{
    prelude::*,
    widgets::*,
};
use chrono::Datelike;
use crate::audit;

/// App state
pub struct App {
    /// Current tab
    tab: usize,
    /// Tab names
    tabs: Vec<&'static str>,
    /// Should quit
    should_quit: bool,
    /// Selected memory index
    selected_memory: usize,
    /// Memory list (content previews)
    memories: Vec<(String, String, String)>, // (id, content, tags)
    /// Stats
    stats: (usize, usize, usize), // stores, recalls, searches
    /// Weekly stats
    weekly_stats: Vec<(String, usize, usize, usize)>,
    /// Scroll offset for memory list
    scroll_offset: usize,
}

impl App {
    pub fn new() -> Self {
        let stats = audit::get_daily_stats();
        let weekly_stats = audit::get_weekly_stats();
        
        Self {
            tab: 0,
            tabs: vec!["📊 Dashboard", "🧠 Memories", "📈 Trends", "🔍 Search"],
            should_quit: false,
            selected_memory: 0,
            memories: Vec::new(),
            stats,
            weekly_stats,
            scroll_offset: 0,
        }
    }
    
    pub fn load_memories(&mut self, memories: Vec<(String, String, String)>) {
        self.memories = memories;
    }
    
    fn next_tab(&mut self) {
        self.tab = (self.tab + 1) % self.tabs.len();
    }
    
    fn prev_tab(&mut self) {
        if self.tab > 0 {
            self.tab -= 1;
        } else {
            self.tab = self.tabs.len() - 1;
        }
    }
    
    fn next_memory(&mut self) {
        if !self.memories.is_empty() {
            self.selected_memory = (self.selected_memory + 1) % self.memories.len();
        }
    }
    
    fn prev_memory(&mut self) {
        if !self.memories.is_empty() {
            if self.selected_memory > 0 {
                self.selected_memory -= 1;
            } else {
                self.selected_memory = self.memories.len() - 1;
            }
        }
    }
}

/// Run the TUI
pub fn run_tui(memories: Vec<(String, String, String)>) -> io::Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
    
    // Create app
    let mut app = App::new();
    app.load_memories(memories);
    
    // Main loop
    loop {
        // Draw
        terminal.draw(|frame| ui(frame, &app))?;
        
        // Handle events
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => app.should_quit = true,
                        KeyCode::Tab | KeyCode::Right => app.next_tab(),
                        KeyCode::BackTab | KeyCode::Left => app.prev_tab(),
                        KeyCode::Down | KeyCode::Char('j') => app.next_memory(),
                        KeyCode::Up | KeyCode::Char('k') => app.prev_memory(),
                        KeyCode::Char('1') => app.tab = 0,
                        KeyCode::Char('2') => app.tab = 1,
                        KeyCode::Char('3') => app.tab = 2,
                        KeyCode::Char('4') => app.tab = 3,
                        _ => {}
                    }
                }
            }
        }
        
        if app.should_quit {
            break;
        }
    }
    
    // Restore terminal
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    
    Ok(())
}

/// Draw the UI
fn ui(frame: &mut Frame, app: &App) {
    let area = frame.area();
    
    // Main layout
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Length(3),  // Tabs
            Constraint::Min(10),    // Content
            Constraint::Length(3),  // Footer
        ])
        .split(area);
    
    // Header
    let header = Paragraph::new("🧠 Memory Brain TUI Dashboard")
        .style(Style::default().fg(Color::Cyan).bold())
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Cyan)));
    frame.render_widget(header, chunks[0]);
    
    // Tabs
    let tabs: Vec<Line> = app.tabs.iter()
        .enumerate()
        .map(|(i, t)| {
            if i == app.tab {
                Line::from(Span::styled(*t, Style::default().fg(Color::Yellow).bold()))
            } else {
                Line::from(Span::styled(*t, Style::default().fg(Color::White)))
            }
        })
        .collect();
    
    let tab_titles: Vec<&str> = app.tabs.to_vec();
    let tabs_widget = Tabs::new(tab_titles)
        .select(app.tab)
        .style(Style::default().fg(Color::White))
        .highlight_style(Style::default().fg(Color::Yellow).bold())
        .divider("|")
        .block(Block::default().borders(Borders::ALL).title(" Tabs [1-4] "));
    frame.render_widget(tabs_widget, chunks[1]);
    
    // Content based on tab
    match app.tab {
        0 => draw_dashboard(frame, chunks[2], app),
        1 => draw_memories(frame, chunks[2], app),
        2 => draw_trends(frame, chunks[2], app),
        3 => draw_search(frame, chunks[2], app),
        _ => {}
    }
    
    // Footer
    let footer = Paragraph::new(" ←/→ or Tab: Switch tabs | ↑/↓ or j/k: Navigate | q: Quit ")
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)));
    frame.render_widget(footer, chunks[3]);
}

/// Draw dashboard tab
fn draw_dashboard(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);
    
    // Left: Today's stats
    let (stores, recalls, searches) = app.stats;
    let total = stores + recalls + searches;
    
    let stats_text = vec![
        Line::from(vec![
            Span::styled("📥 Stores:   ", Style::default().fg(Color::Green)),
            Span::styled(format!("{}", stores), Style::default().fg(Color::Green).bold()),
        ]),
        Line::from(vec![
            Span::styled("🔍 Recalls:  ", Style::default().fg(Color::Blue)),
            Span::styled(format!("{}", recalls), Style::default().fg(Color::Blue).bold()),
        ]),
        Line::from(vec![
            Span::styled("🔎 Searches: ", Style::default().fg(Color::Yellow)),
            Span::styled(format!("{}", searches), Style::default().fg(Color::Yellow).bold()),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("📊 Total:    ", Style::default().fg(Color::White).bold()),
            Span::styled(format!("{}", total), Style::default().fg(Color::Cyan).bold()),
        ]),
    ];
    
    let stats_widget = Paragraph::new(stats_text)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" Today's Activity ")
            .border_style(Style::default().fg(Color::Green)));
    frame.render_widget(stats_widget, chunks[0]);
    
    // Right: Quick info
    let info_text = vec![
        Line::from(vec![
            Span::styled("🧠 Total Memories: ", Style::default()),
            Span::styled(format!("{}", app.memories.len()), Style::default().fg(Color::Cyan).bold()),
        ]),
        Line::from(""),
        Line::from(Span::styled("Quick Keys:", Style::default().fg(Color::Yellow))),
        Line::from("  1-4: Jump to tab"),
        Line::from("  Tab: Next tab"),
        Line::from("  j/k: Navigate list"),
        Line::from("  q: Quit"),
    ];
    
    let info_widget = Paragraph::new(info_text)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" Info ")
            .border_style(Style::default().fg(Color::Magenta)));
    frame.render_widget(info_widget, chunks[1]);
}

/// Draw memories list tab
fn draw_memories(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(area);
    
    // Left: Memory list
    let items: Vec<ListItem> = app.memories.iter()
        .enumerate()
        .map(|(i, (id, content, _))| {
            let preview: String = content.chars().take(30).collect();
            let display = format!("{}: {}...", &id[..8.min(id.len())], preview);
            let style = if i == app.selected_memory {
                Style::default().fg(Color::Yellow).bold()
            } else {
                Style::default().fg(Color::White)
            };
            ListItem::new(display).style(style)
        })
        .collect();
    
    let list = List::new(items)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(format!(" Memories ({}) ", app.memories.len()))
            .border_style(Style::default().fg(Color::Cyan)))
        .highlight_style(Style::default().add_modifier(Modifier::BOLD))
        .highlight_symbol("▶ ");
    
    let mut state = ListState::default();
    state.select(Some(app.selected_memory));
    frame.render_stateful_widget(list, chunks[0], &mut state);
    
    // Right: Selected memory details
    let detail_content = if let Some((id, content, tags)) = app.memories.get(app.selected_memory) {
        vec![
            Line::from(vec![
                Span::styled("ID: ", Style::default().fg(Color::DarkGray)),
                Span::styled(id, Style::default().fg(Color::Cyan)),
            ]),
            Line::from(""),
            Line::from(Span::styled("Content:", Style::default().fg(Color::Yellow))),
            Line::from(content.as_str()),
            Line::from(""),
            Line::from(vec![
                Span::styled("Tags: ", Style::default().fg(Color::DarkGray)),
                Span::styled(tags, Style::default().fg(Color::Green)),
            ]),
        ]
    } else {
        vec![Line::from("No memory selected")]
    };
    
    let detail = Paragraph::new(detail_content)
        .wrap(Wrap { trim: true })
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" Details ")
            .border_style(Style::default().fg(Color::Yellow)));
    frame.render_widget(detail, chunks[1]);
}

/// Draw trends tab
fn draw_trends(frame: &mut Frame, area: Rect, app: &App) {
    // Create bar chart data
    let data: Vec<(&str, u64)> = app.weekly_stats.iter()
        .map(|(date, s, r, se)| {
            // Get day name
            let day = if let Ok(d) = chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d") {
                match d.weekday() {
                    chrono::Weekday::Mon => "Mon",
                    chrono::Weekday::Tue => "Tue",
                    chrono::Weekday::Wed => "Wed",
                    chrono::Weekday::Thu => "Thu",
                    chrono::Weekday::Fri => "Fri",
                    chrono::Weekday::Sat => "Sat",
                    chrono::Weekday::Sun => "Sun",
                }
            } else {
                "???"
            };
            (day, (*s + *r + *se) as u64)
        })
        .collect();
    
    let barchart = BarChart::default()
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" 📈 Weekly Activity ")
            .border_style(Style::default().fg(Color::Magenta)))
        .data(&data)
        .bar_width(7)
        .bar_gap(2)
        .bar_style(Style::default().fg(Color::Cyan))
        .value_style(Style::default().fg(Color::White).bold());
    
    frame.render_widget(barchart, area);
}

/// Draw search tab (placeholder)
fn draw_search(frame: &mut Frame, area: Rect, _app: &App) {
    let text = vec![
        Line::from(""),
        Line::from(Span::styled("🔍 Search (Coming Soon!)", Style::default().fg(Color::Yellow).bold())),
        Line::from(""),
        Line::from("Type to search through memories..."),
        Line::from(""),
        Line::from(Span::styled("Features planned:", Style::default().fg(Color::Cyan))),
        Line::from("  • Full-text search"),
        Line::from("  • Tag filtering"),
        Line::from("  • Date range"),
        Line::from("  • Semantic search"),
    ];
    
    let widget = Paragraph::new(text)
        .alignment(Alignment::Center)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" Search ")
            .border_style(Style::default().fg(Color::Yellow)));
    frame.render_widget(widget, area);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_app_creation() {
        let app = App::new();
        assert_eq!(app.tab, 0);
        assert!(!app.should_quit);
    }
    
    #[test]
    fn test_tab_navigation() {
        let mut app = App::new();
        app.next_tab();
        assert_eq!(app.tab, 1);
        app.prev_tab();
        assert_eq!(app.tab, 0);
    }
}
