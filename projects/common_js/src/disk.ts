export class DiskManager {
  key: string;
  constructor(key: string) {
    this.key = key;
  }

  save(values: object) {
    localStorage.setItem(this.key, JSON.stringify(values));
  }

  reset() {
    localStorage.removeItem(this.key);
  }

  read() {
    const savedData = localStorage.getItem(this.key);
    return savedData ? JSON.parse(savedData) : null;
  }
}
