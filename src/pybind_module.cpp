#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Thompson.hpp"

namespace py = pybind11;

PYBIND11_MODULE(fast_thompson, m) {
    py::class_<Item>(m, "Item")
        .def(py::init<uint32_t, int, int>())
        .def_readwrite("id", &Item::id)
        .def_readwrite("successes", &Item::successes)
        .def_readwrite("failures", &Item::failures)
        .def("isApprox", &Item::isApprox);

    py::class_<IdWithScore>(m, "IdWithScore")
        .def_readonly("id", &IdWithScore::id)
        .def_readonly("score", &IdWithScore::score);

    py::class_<Thompson>(m, "Thompson")
        .def(py::init<const std::vector<Item>&, uint64_t>(),
             py::arg("items"),
             py::arg("seed") = 0xdeadbeefcafebabeULL)
        .def("sample", &Thompson::sample,
             py::arg("num"), py::arg("forbidden"));
}