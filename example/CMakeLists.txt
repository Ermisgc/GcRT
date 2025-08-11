add_executable(importer import.cpp)

add_executable(test test.cpp)
target_link_libraries(importer gcrt)
target_link_libraries(test gcrt)