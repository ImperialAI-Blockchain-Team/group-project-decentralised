import web3 from './web3';
import { modelDatabase} from "../abi/abi";

const address = '0xa3352C250c59850411a81316dE85fBbD85E2D3c5';

export default new web3.eth.Contract(modelDatabase, address);
