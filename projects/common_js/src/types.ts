export interface BalanceSnapshot {
  id: number;
  created_at: string;
  value: number;
  debt: number;
  btc_price: number;
  long_assets_value: number;
  margin_level: number;
  num_directional_positions: number;
  num_ls_positions: number;
}
