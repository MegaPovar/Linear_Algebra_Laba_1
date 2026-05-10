CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -pedantic
TARGET := lab1
SOURCES := main.cpp gauss.cpp lu.cpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SOURCES) gauss.h lu.h utils.h types.h
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
