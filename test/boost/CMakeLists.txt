add_executable(test_boost stimer.cpp)
target_link_libraries(test_boost ${Boost_LIBRARIES})

add_executable(json_test json.cpp)
target_link_libraries(json_test ${Boost_LIBRARIES})

add_executable(beast_test beast_async.cpp)
target_link_libraries(beast_test ${Boost_LIBRARIES})