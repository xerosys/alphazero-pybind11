#include "connect5_gs.h"

#include <cstring>
#include <stdexcept>

namespace alphazero::connect5_gs {

// Layout: 2*7*9 int8 board (126B) | int8 player (1B) | int32 turn (4B) = 131B
static constexpr size_t kSerializedSize = 126 + 1 + 4;

[[nodiscard]] std::unique_ptr<GameState> Connect5GS::copy() const noexcept {
  return std::make_unique<Connect5GS>(board_, player_, turn_);
}

[[nodiscard]] bool Connect5GS::operator==(
    const GameState& other) const noexcept {
  const auto* other_cs = dynamic_cast<const Connect5GS*>(&other);
  if (other_cs == nullptr) {
    return false;
  }
  for (auto p = 0; p < 2; ++p) {
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        if (other_cs->board_(p, h, w) != board_(p, h, w)) {
          return false;
        }
      }
    }
  }
  return (other_cs->player_ == player_);
}

void Connect5GS::hash(absl::HashState h) const {
  h = absl::HashState::combine_contiguous(std::move(h), board_.data(),
                                          board_.size());
  absl::HashState::combine(std::move(h), player_);
}

[[nodiscard]] Vector<uint8_t> Connect5GS::valid_moves() const noexcept {
  auto valids = Vector<uint8_t>{WIDTH};
  for (auto w = 0; w < WIDTH; ++w) {
    valids(w) =
        static_cast<uint8_t>(board_(0, 0, w) == 0 && board_(1, 0, w) == 0);
  }
  return valids;
}

void Connect5GS::play_move(uint32_t move) {
  for (auto h = HEIGHT - 1; h >= 0; --h) {
    if (board_(0, h, move) == 0 && board_(1, h, move) == 0) {
      board_(player_, h, move) = 1;
      player_ = (player_ + 1) % 2;
      ++turn_;
      return;
    }
  }
  throw std::runtime_error{"Invalid move: You have a bug in your code."};
}

[[nodiscard]] std::optional<Vector<float>> Connect5GS::scores() const noexcept {
  auto scores = SizedVector<float, 3>{};
  scores.setZero();
  for (auto p = 0; p < 2; ++p) {
    for (auto h = 0; h < HEIGHT; ++h) {
      auto total = 0;
      for (auto w = 0; w < WIDTH; ++w) {
        if (board_(p, h, w) == 1) {
          ++total;
        } else {
          total = 0;
        }
        if (total == WIN_LENGTH) {
          scores(p) = 1;
          return scores;
        }
      }
    }
    for (auto w = 0; w < WIDTH; ++w) {
      auto total = 0;
      for (auto h = 0; h < HEIGHT; ++h) {
        if (board_(p, h, w) == 1) {
          ++total;
        } else {
          total = 0;
        }
        if (total == WIN_LENGTH) {
          scores(p) = 1;
          return scores;
        }
      }
    }
    for (auto h = 0; h < HEIGHT - WIN_LENGTH + 1; ++h) {
      for (auto w = 0; w < WIDTH - WIN_LENGTH + 1; ++w) {
        auto good = true;
        for (auto x = 0; x < WIN_LENGTH; ++x) {
          if (board_(p, h + x, w + x) == 0) {
            good = false;
            break;
          }
        }
        if (good) {
          scores(p) = 1;
          return scores;
        }
      }
      for (auto w = WIDTH - WIN_LENGTH; w < WIDTH; ++w) {
        auto good = true;
        for (auto x = 0; x < WIN_LENGTH; ++x) {
          if (board_(p, h + x, w - x) == 0) {
            good = false;
            break;
          }
        }
        if (good) {
          scores(p) = 1;
          return scores;
        }
      }
    }
  }
  auto valids = valid_moves();
  for (auto w = 0; w < WIDTH; ++w) {
    if (valids(w) == 1) {
      return std::nullopt;
    }
  }
  scores(2) = 1;
  return scores;
}

[[nodiscard]] Tensor<float, 3> Connect5GS::canonicalized() const noexcept {
  auto out = CanonicalTensor{};
  for (auto p = 0; p < 2; ++p) {
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        out(p, h, w) = board_(p, h, w);
      }
    }
  }
  auto p = player_ + 2;
  auto op = (player_ + 1) % 2 + 2;
  for (auto h = 0; h < HEIGHT; ++h) {
    for (auto w = 0; w < WIDTH; ++w) {
      out(p, h, w) = 1;
      out(op, h, w) = 0;
    }
  }
  return out;
}

[[nodiscard]] std::vector<PlayHistory> Connect5GS::symmetries(
    const PlayHistory& base) const noexcept {
  std::vector<PlayHistory> syms{base};
  PlayHistory mirror;
  mirror.v = base.v;
  mirror.canonical = CanonicalTensor{};
  for (auto f = 0; f < CANONICAL_SHAPE[0]; ++f) {
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        mirror.canonical(f, h, w) = base.canonical(f, h, (WIDTH - 1) - w);
      }
    }
  }
  mirror.pi = Vector<float>{WIDTH};
  for (auto w = 0; w < WIDTH; ++w) {
    mirror.pi(w) = base.pi((WIDTH - 1) - w);
  }
  syms.push_back(mirror);
  return syms;
}

[[nodiscard]] std::string Connect5GS::to_bytes() const {
  std::string out(kSerializedSize, '\0');
  std::memcpy(&out[0], board_.data(), 126);
  out[126] = static_cast<char>(player_);
  std::memcpy(&out[127], &turn_, 4);
  return out;
}

[[nodiscard]] Connect5GS Connect5GS::from_bytes(const std::string& data) {
  if (data.size() != kSerializedSize) {
    throw std::runtime_error("Connect5GS::from_bytes: wrong byte length");
  }
  BoardTensor board{};
  std::memcpy(board.data(), &data[0], 126);
  int8_t player = static_cast<int8_t>(data[126]);
  int32_t turn = 0;
  std::memcpy(&turn, &data[127], 4);
  return Connect5GS(board, player, turn);
}

[[nodiscard]] std::string Connect5GS::dump() const noexcept {
  auto out = "Current Player: " + std::to_string(player_) + '\n';
  for (auto h = 0; h < HEIGHT; ++h) {
    for (auto w = 0; w < WIDTH; ++w) {
      if (board_(0, h, w) == 1) {
        out += 'X';
      } else if (board_(1, h, w) == 1) {
        out += 'O';
      } else {
        out += '.';
      }
    }
    out += '\n';
  }
  out += '\n';
  return out;
}

}  // namespace alphazero::connect5_gs
